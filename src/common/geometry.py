import numpy as np
SQRT_2 = np.sqrt(2)

def vec(x, y, z):
	""" column vector """
	v = np.zeros((3, 1), dtype=np.float64)
	v[0, 0] = x
	v[1, 0] = y
	v[2, 0] = z
	return v

def get_img_size(img):
	return np.array((img.shape[1], img.shape[0]), dtype=np.int32)

def nlize(v):
	""" Normalize vector in place """
	v /= np.linalg.norm(v)
	return v 

def normal_to_vector2d(v):
	return nlize(np.array([-v[1], v[0]]))

ANG_FULL = 2.0*np.pi

def angular_distance_abs(ang_1, ang_2):
	return np.abs(np.mod((ang_1 - ang_2) + np.pi, ANG_FULL) - np.pi)

CORNERS_RW = np.array([
	[0, 0],
	[1, 0],
	[1, 1],
	[0, 1],	
])

def get_img_corners(img):
	imgsz = get_img_size(img)
	return CORNERS_RW * imgsz.reshape(1, 2)

###################################################################################################
# Affine and homography transforms
###################################################################################################

def affine_translation(t):
	M = np.eye(3, dtype=np.float64)
	M[0, 2] = t[0]
	M[1, 2] = t[1]
	return M

def affine_scale_vect(scale_vect):
	return np.diag(scale_vect)

def affine_scale(sx, sy):
	return affine_scale_vect(np.array((sx, sy, 1.)))

def patch_cutting_affine_matrix(patch_center_xy, patch_shape_xy):
	"""
		Produces a pair of intrinsic matrices that homography:
			Kl @ H @ Kr
		cuts out a patch in addition to applying the warping 
	"""
	Kl = affine_translation(np.array(patch_shape_xy)*0.5)
	Kr = affine_translation(-np.array(patch_center_xy))
	return (Kl, Kr)

def extend_with_neutral_row(src_mat):
	"""
		Adds a row filled with 1s at the end
	"""
	return np.vstack((
		src_mat,
		np.ones((1, src_mat.shape[1]), dtype=src_mat.dtype),
	))

def homography_apply(mat, points):
	"""
	@param mat: homography 3x3
	@param points: array of points, points are cols, 2xN
	"""

	if points.shape[0] == 2:
		points = extend_with_neutral_row(points)

	pp = mat @ points
	return pp[:2, :] / pp[2, :]

def homography_apply_rowvec(mat, points_rw):
	"""
	Applies homography H to points
	@param mat: homography 3x3
	@param points_rw: points are rows, Nx2
	"""
	return homography_apply(mat, points_rw.T).T

###################################################################################################
# 3D homogenous transforms
###################################################################################################

def spatial_transform(t = np.zeros((3, 1), dtype=np.float64), r = np.eye(3, dtype=np.float64)):
	"""
		3D sptial transform matrix using homogenous coordinates
		@param t: translation vector, shape (3, 1)
		@param r: rotation matrix, shape (3, 3)
	"""
	mat = np.eye(4, dtype=np.float64)
	mat[:3, 3:4] = t.reshape((3, 1))
	mat[:3, :3] = r.reshape((3, 3))
	return mat

homog_transform = spatial_transform

def spatial_inverse(mat):
	"""
		Inverse a 3D spatial transform matrix
		p2 = R p1 + T
		p1 = R.t @ (p2 - T) = R.t @ p2 + R.t @ (-T)
	"""

	minus_t1 = -mat[:3, 3:4] # negate translation
	r1_T = mat[:3, :3].T # transpose rotation to get inverse

	return spatial_transform(
		t = r1_T @ minus_t1, # + R.t @ (-T)
		r = r1_T, # R.t @ p2
	)

def spatial_transform_apply(mat, points):
	"""
	@param mat: transformation matrix 4x4 or 3x4
	@param points: array of points, points are cols, 3xN or 4xN
	"""

	if points.shape[0] == 3:
		points = extend_with_neutral_row(points)

	return mat @ points

def projection_apply(mat, points):
	""" Project 3D points to camera image plane
	@param mat: projection matrix 4x4 or 3x4
	@param points: array of points, points are cols, 3xN or 4xN
	"""
	pt_img_homog = spatial_transform_apply(mat, points)
	return pt_img_homog[:2, :] / pt_img_homog[2, :]

def projection_apply_rowvec(mat, points):
	""" project 3D points to camera image plane """
	return projection_apply(mat, points.T).T

def rot_around_x(angle):
	""" rotation matrix around x axis """
	s = np.sin(angle)
	c = np.cos(angle)

	return np.array([
		[1, 0, 0],
		[0, c, s],
		[0, -s, c],
	], dtype=np.float64)

def rot_around_y(angle):
	""" rotation matrix around y axis """
	s = np.sin(angle)
	c = np.cos(angle)

	return np.array([
		[c, 0, -s],
		[0, 1, 0],
		[s, 0, c],
	], dtype=np.float64)

def rot_around_z(angle):
	""" rotation matrix around z axis """
	s = np.sin(angle)
	c = np.cos(angle)

	return np.array([
		[c, -s, 0],
		[s, c, 0],
		[0, 0, 1],
	], dtype=np.float64)


###################################################################################################
# Camera intrinsics
###################################################################################################

def intrinsic_matrix(focal, size_xy):
	intrinsic_mat = np.eye(3, dtype=np.float64)
	intrinsic_mat[0, 0] = focal
	intrinsic_mat[1, 1] = focal
	intrinsic_mat[0, 2] = size_xy[0] / 2
	intrinsic_mat[1, 2] = size_xy[1] / 2
	return intrinsic_mat

def patch_cutting_intrinsic_matrix(focal, patch_center_xy, patch_shape_xy):
	"""
		Produces a pair of intrinsic matrices that homography:
			Kl @ H @ Kr
		cuts out a patch in addition to applying the warping 
	"""

	intrinsic_mat = np.eye(3, dtype=np.float64)
	intrinsic_mat[0, 0] = focal
	intrinsic_mat[1, 1] = focal
	intrinsic_mat[0, 2] = patch_center_xy[0]
	intrinsic_mat[1, 2] = patch_center_xy[1]

	Kr = np.linalg.inv(intrinsic_mat)
	
	intrinsic_mat[0, 2] = patch_shape_xy[0] / 2
	intrinsic_mat[1, 2] = patch_shape_xy[1] / 2
	
	Kl = intrinsic_mat
	
	return (Kl, Kr)



def normal_to_both(a, b):
	"""
		@return: vector normal to a and b, of length 1
	"""
	cr = np.cross(a.ravel(), b.ravel())
	cr /= np.linalg.norm(cr)
	return cr

# def rot_btw_vectors(A, B):
# 	cross_AB = np.cross(A.T, B.T) # length is sin(angle A B)
# 	sin_AB = np.linalg.norm(cross_AB)
	
# 	if np.abs(sin_AB) < np.spacing(sin_AB):
# 		return np.identity(3)
	
# 	dot_AB = A.T @ B # = cos(angle A B)
# 	cos_AB = dot_AB
	
# 	# normalized rejection of B wrt A
# 	rejection_B_wrt_A = B - A*dot_AB
# 	rejection_B_wrt_A /= np.linalg.norm(rejection_B_wrt_A)
	
# 	# build basis in which we can easily rotate 
# 	#  A 
# 	#  rej B A
# 	#  B cross A = - A cross B
# 	# to get an orthogonal basis, we need to normalize the cross product
# 	rot_basis = np.hstack((A, rejection_B_wrt_A, -cross_AB.T / sin_AB))
		
# 	# rotation in this basis
# 	rot_simple = np.array([
# 		[cos_AB, -sin_AB, 0],
# 		[sin_AB, cos_AB, 0],
# 		[0, 0, 1],
# 	], dtype=np.float64)
	
# 	return rot_basis @ rot_simple @ rot_basis.T