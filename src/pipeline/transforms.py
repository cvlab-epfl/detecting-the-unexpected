import logging
log = logging.getLogger('exp')
from .frame import *
import numpy as np
import torch
import inspect, types

class TrBase:
	""" Base class for transforms, provides a __repr__ """
	
	@staticmethod
	def object_name(obj):
		return getattr(obj, '__name__', None) or obj.__class__.__name__
	
	@staticmethod
	def repr_with_args(name, args):
		return '{name}({args})'.format(
			name = name,
			args = ', '.join(args)
		)
	
	@classmethod
	def callable_signature_to_string(cls, callable_obj, objname=None):
		sig = inspect.signature(callable_obj)
		objname = objname or cls.object_name(callable_obj)
		
		return cls.repr_with_args(
			objname, 
			args = [
				arg.name + ('?' if arg.default != arg.empty else '')
				for arg in sig.parameters.values()
				if arg.kind != arg.VAR_KEYWORD
			],
		)
	
	def __repr__(self):
		return self.callable_signature_to_string(self, self.object_name(self))


class TrsChain(list):
	def __init__(self, *transform_list):
		# TrsChain([f1, f2, f3])
		if transform_list.__len__() == 1 and isinstance(transform_list[0], list):
			super().__init__(transform_list[0])
		# TrsChain(f1, f2, f3)
		else:
			super().__init__(transform_list)

	def __call__(self, frame, **_):
		for tr in self:
			if tr is not None:
				frame.apply(tr)

	@staticmethod
	def transform_name(tr):
		if isinstance(tr, types.FunctionType) or isinstance(tr, types.MethodType):
			return TrBase.callable_signature_to_string(tr)
		elif isinstance(tr, torch.nn.Module):
			return '<module> ' + TrBase.callable_signature_to_string(tr)
		else:
			repr = str(tr)
			if repr.__len__() > 87:
				repr =  repr[:87] + '...'

			repr = repr.replace('\n', ' ')
			return repr

	def __repr__(self):
		self_name = self.__class__.__name__

		if self:
			return self_name + '(\n	' + '\n	'.join(map(self.transform_name, self)) + '\n)'
		else:
			return self_name + '()'

	def __add__(self, other):
		return TrsChain(super().__add__(other))

	def copy(self):
		return TrsChain(super().copy())

class TrByField(TrBase):
	"""
	A transform which applies a 1-to-1 function on selected fields
	Example
		TrByField([(img, img_tr)], torch.from_numpy)	
	"""
	
	def __init__(self, fields='*', operation=None):
		"""
		:param fields: list of fields to which the function is applied, an element is:
			a string - frame[field] = func(frame[field])
			a tuple (in, out) - frame[out] = func(frame[in])
			Alternatively fields=='*' means apply to all fields
		:param operation: function to apply on those fields
			alternatively, extend this class and override self.forward
		"""
	
		if fields == '*':
		# TrByField('*')
			self.fields_all = True
		elif isinstance(fields, str):
		# TrByField('field')
			self.fields_all = False
			self.field_pairs = [(fields, fields)]
		elif isinstance(fields, dict):
		# TrByField({'f1in': 'f1out', 'f2in': 'f2out'})
			self.fields_all = False

			self.field_pairs = list(fields.items())
		else:
		# TrByField(['f1', (f1, f2)])
			self.fields_all = False

			self.field_pairs = [
				f if isinstance(f, tuple) or isinstance(f, list) else (f, f)
				for f in fields
			]
	
		self.operation = operation
	
	def forward(self, field_name, value):
		if self.operation is not None:
			return self.operation(value)
		else:
			raise NotImplementedError("TrByField::forward")
	
	def conditionally_complain_about_type(self, field_name, value, should_be):
		if not self.fields_all:
			# only complain if the field was specifically requested
			log.warning(f'{self} request for field {field_name} which is a {type(value)} and not a {should_be}')
	
	def __call__(self, frame, **_):
		if not self.fields_all:
			return {
				fi_out: self.forward(fi_in, frame[fi_in])
				for(fi_in, fi_out) in self.field_pairs
			}
		else:
			return {
				fi: self.forward(fi, val)
				for fi, val in frame.items()
			}

	def __repr__(self):
		return self.repr_with_args(
			name = '{name}{op}'.format(
				name = TrBase.object_name(self),
				op = ('<' + TrBase.object_name(self.operation) + '>') if self.operation is not None else '',
			),
			args = '*' if self.fields_all else [
				fp[0] if fp[0] == fp[1] else (fp[0] + ' -> ' + fp[1])
				for fp in self.field_pairs
			],
		)

def tr_print(frame, **_):
	log.info(frame)

class TrPrint(TrBase):
	def __init__(self, message):
		self.message = message
	
	def __call__(self, frame, **_):
		log.info(self.message, frame)

class TrNoOp(TrBase):
	def __call__(self, **_):
		return None

class TrRenameKw(TrBase):
	def __init__(self, *args, b_copy=False, **field_dict):
		if args.__len__() == 1:
			a0 = args[0]
			if isinstance(a0, list):
				self.field_pairs = a0
			else:
				self.field_pairs = args[0].items()

		elif args.__len__() == 0:
			self.field_pairs = field_dict.items()
		else:
			raise Exception("Argument should be TrRenameKw({'a': 'b'}) or TrRenameKw(a = 'b')")

		self.b_copy = b_copy

	def __call__(self, frame, **_):
		return {
			new_name: frame[old_name] if self.b_copy else frame.pop(old_name)
			for (old_name, new_name) in self.field_pairs
		}

	def __repr__(self):
		return self.repr_with_args(
			self.object_name(self),
			(fp[0] + ' -> ' + fp[1] for fp in self.field_pairs),
		)


TrRename = TrRenameKw

class TrCopy(TrRenameKw):
	def __init__(self, *args, b_copy=True, **kwargs):
		super().__init__(*args, b_copy=b_copy, **kwargs)


class TrRemoveFields(TrBase):
	def __init__(self, *fields):
		# T([f1, f2, f3])
		if fields.__len__() == 1 and isinstance(fields[0], list):
			self.fields = set(fields[0])
		# T(f1, f2, f3)
		else:
			self.fields = set(fields)

	def __call__(self, frame, **_):
		for field in self.fields:
			del frame[field]

	def __repr__(self):
		return self.repr_with_args(self.object_name(self), self.fields)


class TrKeepFields(TrBase):
	def __init__(self, *fields):
		# T([f1, f2, f3])
		if fields.__len__() == 1 and isinstance(fields[0], list):
			self.fields = set(fields[0])
		# T(f1, f2, f3)
		else:
			self.fields = set(fields)

	def __call__(self, frame, **_):
		existing_fields = set(frame.keys())
		missing = self.fields - existing_fields
		if missing:
			log.error(f'TrKeepFields: missing fields {missing}')

		to_remove = existing_fields - self.fields

		for field in to_remove:
			del frame[field]
			
	def __repr__(self):
		return self.repr_with_args(self.object_name(self), self.fields)

class TrKeepFieldsByPrefix(TrBase):
	def __init__(self, *prefixes):
		# T([f1, f2, f3])
		if prefixes.__len__() == 1 and isinstance(prefixes[0], list):
			self.prefixes = prefixes[0]
		# T(f1, f2, f3)
		else:
			self.prefixes = prefixes
	
	def should_field_be_kept(self, field):
		for p in self.prefixes:
			if field.startswith(p):
				return True
		return False
	
	def __call__(self, frame, **_):
		to_keep = set(k for k in frame.keys() if self.should_field_be_kept(k))
		to_remove = set(frame.keys()) - to_keep
		for field in to_remove:
			del frame[field]
			
	def __repr__(self):
		return self.repr_with_args(self.object_name(self), [p+'*' for p in self.prefixes])



class TrAsType(TrBase):
	def __init__(self, name_to_type):
		self.name_to_type = name_to_type

	@staticmethod
	def convert(val, type):
		if isinstance(val, np.ndarray):
			return val.astype(type)
		if isinstance(val, torch.Tensor):
			return val.type(type)

		raise NotImplementedError('Neither an ndarray nor a torch.Tensor')

	def __call__(self, **fields):
		return {
			name: self.convert(fields[name], tp)
			for name, tp in self.name_to_type.items()
		}

	def __repr__(self):
		return self.repr_with_args(self.object_name(self), [
			'{n}->{t}'.format(n=name, t=tp)
			for name, tp in self.name_to_type.items()
		])


# ---------------------------------------------------------------------
# New Transforms
# frame = tr(frame) instaed of frame.apply(tr)
# Use @FrameTransform on the transform functions to make them fit this syntax
# ---------------------------------------------------------------------

class FrameTransform:
	def __init__(self, func):
		self.func = func

		#TODO functools.update_wrapper

		#TODO could analyze the function's signature here and

	def __call__(self, frame):
		result_overrides = self.func(frame=frame, **frame)
		if result_overrides is not None:
			frame.update(result_overrides)
		return frame

	def __repr__(self):
		return TrBase.callable_signature_to_string(self.func, TrBase.object_name(self.func))

	def partial(self, *args, **kwargs):
		""" Wrap the internal function with a functools.partial """
		return FrameTransform(partial(self.func, *args, **kwargs))

class NtrByField_Impl(TrByField):
	def __call__(self, frame):
		if not self.fields_all:
			override = {
				fi_out: self.forward(fi_in, frame[fi_in])
				for(fi_in, fi_out) in self.field_pairs
			}
		else:
			override = {
				fi: self.forward(fi, val)
				for fi, val in frame.items()
			}

		frame.update(override)
		return frame

def NtrByField(fields):
	return (lambda operation: NtrByField_Impl(fields=fields, operation=operation))

class NTrChain(TrsChain):
	def __call__(self, frame):
		for tr in self:
			frame = tr(frame)

		return frame
