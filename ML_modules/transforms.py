import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, features):
        """ 
        Combine multiple augmentation processes and apply them sequentially.
        
        Parameters
        ----------
        features : Nx6
            initial input features
            
        Returns
        -------
        features : Nx6 
            transformed features
        """
        for t in self.transforms:
            features = t(features)

        return features

class ToTensor(object):

    @staticmethod
    def __call__(features):
        """ 
        Convert numpy.ndarray features to torch.Tensor features.
        
        Parameters
        ----------
        features : Nx6 : obj : 'numpy.ndarray'
            initial input features
            
        Returns
        -------
        features : Nx6 : obj : 'torch.Tensor'
            converted torch.Tensor features
        """
        features = torch.from_numpy(features)
        if not isinstance(features, torch.FloatTensor):
            features = features.float()

        return features

class RandomRotate(object):
    def __init__(self, angle=[1, 1, 1]):
        self.angle = angle

    def __call__(self, features):
        """ 
        Randomly rotate an input.
        
        Parameters
        ----------
        angle : 1X3 : obj : `list`
            list of paramters to control random rotation in each axis 
        features : Nx6 : obj : 'numpy.ndarray'
            initial input features
            
        Returns
        -------
        features : Nx6 : obj : 'numpy.ndarray'
            transformed features
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(features[:,:3])

        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        RM = R.from_euler('xyz', [angle_x, angle_y, angle_z]).as_matrix()
        pcd.rotate(R=RM, center=(0, 0, 0))

        features[:,:3] = np.asarray(pcd.points)

        return features
    
class RandomScale(object):
    def __init__(self, scale=[0.97, 1.03], anisotropic=False):
        self.scale = scale
        self.anisotropic = anisotropic

    def __call__(self, features):
        """ 
        Randomly scale an input.
        
        Parameters
        ----------
        scale : 1X2 : obj : `list`
            min & max  scaling factors
        anisotropic : bool
            whether equally scale an input in every dimension
        features : Nx6 : obj : 'numpy.ndarray'
            initial input features
            
        Returns
        -------
        features : Nx6 : obj : 'numpy.ndarray'
            transformed features
        """
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        features[:,:3] *= scale

        return features
    
class RandomPermute(object):

    @staticmethod
    def __call__(features):
        """ 
        Randomly rotate an input.
        
        Parameters
        ----------
        features : Nx6 : obj : 'numpy.ndarray'
            initial input features
            
        Returns
        -------
        features : Nx6 : obj : 'numpy.ndarray'
            transformed features
        """
        features = features[np.random.permutation(features.shape[0]), :]

        return features

class RandomJitter(object):
    def __init__(self, sigma=0.04, clip=0.07):
        self.sigma, self.clip = sigma, clip

    def __call__(self, features):
        """ 
        Add jitters to the given point cloud features.
        
        Parameters
        ----------
        sigma : float
            standard deviation of a sampling distribution
        clip : float
            cliping value of randomly generated jitters
        features : Nx6 : obj : 'numpy.ndarray'
            initial input features
            
        Returns
        -------
        features : Nx6 : obj : 'numpy.ndarray'
            features with jitters
        """
        assert (self.clip > 0)

        jitter = np.clip(self.sigma * np.random.randn(features.shape[0], 3), -1 * self.clip, self.clip)
        features[:,:3] += jitter

        return features
