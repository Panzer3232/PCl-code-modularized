import dataset_h36m
import dataset_3dhp
import constants


class DatasetConfig:
    def get_joint_mean(self):
        raise NotImplementedError

    def get_joint_std(self):
        raise NotImplementedError

    def get_2d_mean_std(self, slant=False, stn=False, use_pcl=True):
        raise NotImplementedError

    def to_canonical(self, joints):
        raise NotImplementedError

<<<<<<< HEAD

class H36mConfig(DatasetConfig):
=======
    def get_num_joints(self, use_canonical=False):
        raise NotImplementedError

    def get_mean_std_normalized(self, device, slant=False, stn=False, use_pcl=True):
        mean, std = self.get_2d_mean_std(slant=slant, stn=stn, use_pcl=use_pcl)
        return mean.to(device), std.to(device)


class H36mConfig(DatasetConfig):
    def __init__(self):
        self.name = "H36m"

>>>>>>> 82eba86 (Initial commit: Docker + Makefile + training + tests+ codefiles)
    def get_joint_mean(self):
        return dataset_h36m.h36m_to_canonical_skeleton(constants.H36mMean)

    def get_joint_std(self):
        return dataset_h36m.h36m_to_canonical_skeleton(constants.H36mStd)

    def get_2d_mean_std(self, slant=False, stn=False, use_pcl=True):
        """
<<<<<<< HEAD
        Returns 2D joint mean and std in canonical order.
=======
        Returns 2D joint mean and std for MPI-INF-3DHP.
        Supports all 2D normalization variants.
>>>>>>> 82eba86 (Initial commit: Docker + Makefile + training + tests+ codefiles)
        """
        if use_pcl:
            if slant:
                mean = constants.H36m_2d_PCL_Mean_2dScale
                std = constants.H36m_2d_PCL_Std_2dScale
            else:
                mean = constants.H36m_2d_PCL_Mean
                std = constants.H36m_2d_PCL_Std
<<<<<<< HEAD
            return mean, std
        else:
            if slant:
=======
        else:
            if stn:
>>>>>>> 82eba86 (Initial commit: Docker + Makefile + training + tests+ codefiles)
                mean = constants.H36m_2d_STN_Mean_2dScale
                std = constants.H36m_2d_STN_Std_2dScale
            else:
                mean = constants.H36m_2d_Mean
                std = constants.H36m_2d_Std
<<<<<<< HEAD
            return self.to_canonical(mean), self.to_canonical(std)

=======
            mean = self.to_canonical(mean)
            std = self.to_canonical(std)
        return mean, std
>>>>>>> 82eba86 (Initial commit: Docker + Makefile + training + tests+ codefiles)

    def to_canonical(self, joints):
        """
        Reorders joints to canonical order (17 joints) using h36m mapping.
        """
        return dataset_h36m.h36m_to_canonical_skeleton(joints)

<<<<<<< HEAD

class MPI3DHPConfig(DatasetConfig):
=======
    def get_num_joints(self, use_canonical=False):
        return 17 if use_canonical else 32


class MPI3DHPConfig(DatasetConfig):
    def __init__(self):
        self.name = "3DHP"

>>>>>>> 82eba86 (Initial commit: Docker + Makefile + training + tests+ codefiles)
    def get_joint_mean(self):
        return constants.mpi_3d_Mean

    def get_joint_std(self):
        return constants.mpi_3d_Std

    def get_2d_mean_std(self, slant=False, stn=False, use_pcl=True):
        """
        Returns 2D joint mean and std for MPI-INF-3DHP.
        Supports all 2D normalization variants.
        """
        if use_pcl:
            if slant:
                return constants.mpi_2d_pcl_slant_mean, constants.mpi_2d_pcl_slant_std
            else:
                return constants.mpi_2d_pcl_3dscale_mean, constants.mpi_2d_pcl_3dscale_std
<<<<<<< HEAD
        else:  
            if slant:
=======
        else:
            if stn:
>>>>>>> 82eba86 (Initial commit: Docker + Makefile + training + tests+ codefiles)
                return constants.mpi_2d_stn_slant_mean, constants.mpi_2d_stn_slant_std
            else:
                return constants.mpi_2d_stn_3dscale_mean, constants.mpi_2d_stn_3dscale_std

    def to_canonical(self, joints):
<<<<<<< HEAD
        """
        No reordering needed for 3DHP — canonical.
        """
        return joints

=======
        return joints

    def get_num_joints(self, use_canonical=False):
        return 17

>>>>>>> 82eba86 (Initial commit: Docker + Makefile + training + tests+ codefiles)

def get_dataset_config(name):
    """
    Factory function to get the correct dataset config.
    """
    if name == "H36m":
        return H36mConfig()
    elif name == "3DHP":
        return MPI3DHPConfig()
    else:
        raise ValueError(f"Unsupported dataset: {name}")
