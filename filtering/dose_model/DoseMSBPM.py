import torch


class DoseMSBPM(torch.nn.Module):
    """
    Base class for the dlw model.
    """
    def __init__(self, rho_0_GT: torch.Tensor, intensity_nonlinear: torch.Tensor,
                 print_params: list, nonlinearity: torch.Tensor,
                 device='cpu', dtype=torch.float64):
        """

        Args:
            rho_0: Initial structure/power lines [batch_size, 1, nx, ny, nz]
            intensity_nonlinear: Point Spread Function intensity profile
            print_params: (sig_2_r, time_exposure, intensity_without_power, correction_factor) List the parameters
                            relevant to the print.
            nonlinearity: Power exponent of the nonlinearity.
            device: Device which the simulation is run on, options: cuda or cpu.
            dtype: Datatype, recommended torch.float64
        """
        self.rho_0_GT = rho_0_GT.to(device).to(dtype)
        self.intensity_nonlinear = intensity_nonlinear.to(device).to(dtype)

        self.nonlinearity = nonlinearity.to(device).to(dtype)
        self.factor_in_exp = print_params[0] * print_params[1] * (
                print_params[2] ** nonlinearity).to(device).to(dtype)
        self.correction_factor = print_params[3]

        super().__init__()


class DoseMSBPMFull3D(DoseMSBPM):
    '''
    Defines the convolution of hte density with the PSF.
    '''
    def forward(self, obj: torch.Tensor,
                lp: torch.Tensor = torch.tensor([[0.020]])) -> torch.Tensor:
        '''
        Convolution of the density with the PSF. Returns the accumulated power of the structure.
        :param obj: Input density.
        :param lp: Laser Power.
        :return: Accumulated power in the resist as a density.
        '''
        conv = torch.nn.functional.conv3d(obj[None, None], self.intensity_nonlinear, padding='same')
        rho = self.rho_0_GT * (1 - torch.exp(
            - self.factor_in_exp * self.correction_factor * conv * lp[:, :, None, None, None] ** self.nonlinearity))

        return rho
