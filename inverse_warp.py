from __future__ import division
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
pixel_coords = None
'''
    大部分是opencv 包含的一些utils
'''

def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W] 常量定义，一直不变,完全可以不定义，但是为了可读性
    #其中pixel_coords[0][0][i][j]是像素(i,j)的相机坐标x分量
    #其中pixel_coords[0][1][i][j]是像素(i,j)的相机坐标y分量
    #其中pixel_coords[0][2][i][j]是像素(i,j)的相机坐标z分量,但是由于没有注入深度信息， 暂时全1

def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):#像素坐标2相机坐标
    global pixel_coords#1,3,128,416  这个3不是通道，是坐标， lossfunc 在外层正对每个ref-img遍历
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)#make pixel_coords, 全局变量可改变
    #tmp1=pixel_coords[:,:,:h,:w]
    #tmp2=tmp1.expand(b,3,h,w)
    #tmp3=tmp2.reshape(b, 3, -1)#[]
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).reshape(b, 3, -1)# [B, 3, H*W]
    TMP=intrinsics_inv @ current_pixel_coords#投影矩阵逆乘,得到camcoords
    #[B,3,3]@[B,3,H*W]=[B,3,H*W]#大概乘法逻辑懂了，
    #到目前为止, TMp完全没用到深度信息, 这里只是给坐标编码!
    #@符号是矩阵想成，要求两个tensor order一样，并且后两介满足惩罚要求，并且除了后两阶其余维度完全一致
    cam_coords = (TMP).reshape(b, 3, h, w)
    """
    print(pixel_coords[0,:,0,0])#value: 0,0,1
    print(pixel_coords[0,:,1,0])#1,0,1
    print(pixel_coords[0,:,0,1])#0,1,1
    print(pixel_coords[0,:,1,1])#1,1,1

    print(cam_coords[0,:,0,0])#-.8650,-.2286
    print(cam_coords[0,:,0,1])#-.8612,-.2286
    print(cam_coords[0,:,1,0])#-.8650,-.2249
    print(cam_coords[0,:,1,1])#-.8612,-.2249
    
    """
    #cam_coords:B,3,128,416
    #\hat{D}_t(p_t)K^{-1}p_t
    return  depth.unsqueeze(1)*cam_coords #[B,3,H,W]X[B,1,H,W]=[B,3,H,W],2d to 3d 再添加深度信息， 彻底变成真cam_coords



def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]??明明是[B,3,H,W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()

    """
    print(cam_coords[0,:,0,0])
    print(cam_coords[0,:,0,1])
    print(cam_coords[0,:,1,0])
    print(cam_coords[0,:,1,1])

    
    tensor([-0.1386, -0.0366,  0.1602], device='cuda:0', grad_fn=<SelectBackward>)
    tensor([-0.1193, -0.0317,  0.1386], device='cuda:0', grad_fn=<SelectBackward>)
    tensor([-0.1298, -0.0338,  0.1501], device='cuda:0', grad_fn=<SelectBackward>)
    tensor([-0.1263, -0.0330,  0.1466], device='cuda:0', grad_fn=<SelectBackward>)
    已经变得立体了
    """
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]#这个意思：本来每个格子有三个值，所有格子组成一个矩阵，现在吧矩阵拉成一条array操作
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W] Rp+t
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)
    """
    >>> a = torch.randn(4)
    >>> a
    tensor([-1.7120,  0.1734, -0.0478, -0.0922])
    >>> torch.clamp(a, min=-0.5, max=0.5)
    tensor([-0.5000,  0.1734, -0.0478, -0.0922])
    
    """

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b,h,w,2)#变成比较容易理解的二维矩阵，每个格子两个值-,x,y


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]#没齐次化，少了最后一行0001
    return transform_mat


def inverse_warp(img, depth, pose, intrinsics, rotation_mode='euler', padding_mode='zeros'):
    """#比较重要的一个函数,本函数所有的img都是一张ref/src img
    Inverse warp a source image to the target image plane.# article [8]

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]#B: batch-size; 3:channels;H:height;W:width
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')#本来3x4,但是第四列有一个基线参数，被去掉了，成了3x3

    batch_size, _, img_height, img_width = img.size()

    #step1:#\hat{D}_t(p_t)K^{-1}p_t, pixel_coords也使用了(这是个常量！量等于坐标)， 通过深度图和像素坐标，得到相机坐标DIBRZ
    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]2

    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    #论文的K\hat{T}_{t->t+1}
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]#内参x外参

    rot, tr = proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:]#得到R,t ，这里只是为了简化，R[B,3,3], t[b,3,1]
    #p_{t+1} = K\hat{T}_{t->t+1} \hat{D}_t(p_t)K^{-1}p_t
    src_pixel_coords = cam2pixel(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    #这个矩阵的坐标就是

                                 #[B,3,H,W]          [B,H,W,2] 图像偏移变换,src_pixel_coords这里意义就是偏移量？
    projected_img = F.grid_sample(input=img, grid = src_pixel_coords, padding_mode=padding_mode)#[B,3,H,W]

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
#    tes=projected_img.cpu().data.numpy()[0]
 #   plt.imshow(tes)
 #   plt.show()
    return projected_img, valid_points
