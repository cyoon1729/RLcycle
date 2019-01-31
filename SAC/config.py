class Config:
    
    def __init__(self, batch_size, gamma, mean_lambda, std_lambda, z_lambda, soft_tau):
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.mean_lambda = mean_lambda
        self.std_lambda = std_lambda
        self.z_lambda = z_lambda
        self.soft_tau = soft_tau