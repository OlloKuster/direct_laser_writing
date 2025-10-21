import torch


class DoseMSBPM(torch.nn.Module):
    def __init__(self, rho_0: torch.Tensor, intensity_nonlinear: torch.Tensor,
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
        self.rho_0 = rho_0.to(device).to(dtype)
        self.intensity_nonlinear = intensity_nonlinear.to(device).to(dtype)

        self.nonlinearity = nonlinearity.to(device).to(dtype)
        self.factor_in_exp = print_params[0] * print_params[1] * (
                print_params[2] ** nonlinearity).to(device).to(dtype)
        self.correction_factor = print_params[3]

        super().__init__()


class DoseMSBPMFull3D(DoseMSBPM):
    def forward(self, obj: torch.Tensor,
                lp: torch.Tensor = torch.tensor([[0.020]])) -> torch.Tensor:
        conv = torch.nn.functional.conv3d(obj[None, None], self.intensity_nonlinear, padding='same')
        rho = self.rho_0_GT * (1 - torch.exp(
            - self.factor_in_exp * self.correction_factor * conv * lp[:, :, None, None, None] ** self.nonlinearity))

        # rescale to rho_0_GT, divide by rho_th_GT

        rho = rho / rho_th_GT


        return rho.squeeze()
