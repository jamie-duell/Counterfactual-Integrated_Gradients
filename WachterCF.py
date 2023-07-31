class WachterCF(ExplainerBase):

    def __init__(self, data_interface, model_interface):

        super().__init__(data_interface, model_interface)

    def generate_counterfactuals(self, query_instance, features_to_vary, target = 0.5, feature_weights = None, _lambda = 10,
            optimizer = "adam", lr = 0.01, max_iter = 100):
       

        query_instance = torch.FloatTensor(query_instance)

        self._lambda = _lambda

        if feature_weights == None:
            feature_weights = torch.ones(query_instance.shape[1])
        else:
            feature_weights = torch.ones(query_instance.shape[0])
            feature_weights = torch.FloatTensor(feature_weights)

        cf_initialize = torch.rand(query_instance.shape)

        cf_initialize = torch.FloatTensor(cf_initialize)
        cf_initialize = cf_initialize * query_instance
        
        if optimizer == "adam":
            optim = torch.optim.Adam([cf_initialize], lr)
        else:
            optim = torch.optim.RMSprop([cf_initialize], lr)

        for i in range(max_iter):
            cf_initialize.requires_grad = True
            optim.zero_grad()
            loss = self.compute_loss(cf_initialize, query_instance, target)
            loss.backward()
            cf_initialize.grad = cf_initialize.grad * query_instance
            optim.step()
            
            cf_initialize.detach_()


        return cf_initialize
    
    def compute_loss(self, cf_initialize, query_instance, target):
        
        loss2 = torch.sum((cf_initialize - query_instance)**2)
        
        return self._lambda * loss2
