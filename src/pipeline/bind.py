
class bind:
	"""
	The purpose of Frame and pipelines is to disentangle control flow from data.
	We always run the NN and then the loss, but the actual values passed between them may differ.

	Syntax
		bind(func, labels='labels_trainId').outs(partitions='labels_partition')
	results in
		r = func(labels = input['labels_trainId'])
		output['labels_partition'] = r['parititions']

		bind(torch.sigmoid, 'pred_logits').outs('pred_probs')

	Maybe better syntax:
	bind(func, default_1 = x, default_2 = y)
		.ins(labels = 'labels_trainId')
		.outs(partitions='labels_partition')
	
	bind(torch.sigmoid).ins('pred_logits').outs('pred_probs')

	"""

	output_single_name : str = None
	input_single_name : str = None
	output_bindings = None
	input_bindings = None
	default_args = {}

	@staticmethod
	def args_kwargs_to_bindings(args, kwargs):
		return dict(
			**kwargs,
			**{
				a: a for a in args
			}
		)
		

	def __init__(self, func, *args, **kwargs):
		
		self.func = func
		# self.args = args
		# self.kwargs = kwargs

		if args.__len__() == 1 and not kwargs:
			self.input_single_name = args[0]
		else:
			self.input_bindings = self.args_kwargs_to_bindings(args, kwargs)

		# TODO check that input field names are string
		# TODO compare bound inputs against func signature, warn if arguments not provided
	
	def outs(self, *args, **kwargs):

		if args.__len__() == 1 and not kwargs:
			self.output_single_name = args[0]
		else:
			self.output_bindings = self.args_kwargs_to_bindings(args, kwargs)

		return self

	def defaults(self, **kwargs):
		self.default_args = kwargs
		return self

	def __call__(self, **fields):
		if self.input_single_name:
			result = self.func(fields[self.input_single_name], **self.defaults)
		else:
			result = self.func(**{
				arg_name: fields[named_field] 
		 		for (arg_name, named_field) in self.input_bindings.items()
			}, **self.default_args)

		# result = self.func(
		# 	*(
		# 		fields[pos_field] 
		# 		for pos_field in self.args
		# 	),
		# 	**{
		# 		arg_name: fields[named_field] 
		# 		for (arg_name, named_field) in self.kwargs.items()
		# 	},
		# )
	
		if self.output_single_name:
			return {self.output_single_name: result}
		
		elif result is not None:
			if not isinstance(result, dict):
				raise ValueError(f'Function return is not dict but {type(result)} but there is no single output name, tr={self}')

			# no bindings, pass on names returned by function
			if self.output_bindings is None:
				return result

			return {
				bound_out_name: result[func_out_name]
				for (func_out_name, bound_out_name) in self.output_bindings.items()
			}

	def __repr__(self):
		return f'bind({self.func})'

