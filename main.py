from src.core_manager import CoreManager


if __name__ == '__main__':
    simulation_parameters = {
        'target': 'fluid',
        'case': 3,
        'resolution': (100, 100, 100),
        'dt': 1.0,
        'dx': 1.0,
        'tau': 1.0,
        'Re': 100.0,
        'total_steps': 10000,
        'make_video': True,
    }
    case_parameters = [
        {
            'name': 'Taylor-Green-Vortex',
            'method': 'LBM_2D',
            'image': "velocity",
            'lambda_alpha': 1.0
        },
        {
            'case_name': 'doubly periodic shear layer',
            'method': 'LBM_2D_KBC',
            "image": "vort",
            'sigma': 0.05,
            'kappa': 80.0
        },
        {
            'case_name': "A liquid cube -> A liquid sphere",
            'method': 'LBM_2D_SC',
            "image": "rho",
            "rho_liquid": 0.265,
            "rho_gas": 0.038,
            "rho_wall": 1.0
        },
        {
            'case_name': "A liquid cube -> A liquid sphere (3D)",
            'method': 'LBM_3D_SC',
            "image": "rho",
            "rho_liquid": 0.265,
            "rho_gas": 0.038,
            "rho_wall": 1.0
        },
        {
            'case_name': "A liquid cube -> A liquid sphere (3D) (HCZ Model)",
            'method': 'LBM_3D_HCZ',
            "image": "rho",
            "psi_max": 0.265,
            "psi_min": 0.038,
            "density_liquid": 0.265,
            "density_gas": 0.038,
            "rho_liquid": 0.265,
            "rho_gas": 0.038,
            "kappa": 0.01,
            "tau_f": 0.7,
            "tau_g": 0.7,
            "theta_A": 70.0,
            "theta_R": 70.0,
            "wall": ["      "]
        }
    ]
    core_manager = CoreManager(simulation_parameters, case_parameters)
    core_manager.initialization()
    core_manager.run()
    core_manager.kill()
