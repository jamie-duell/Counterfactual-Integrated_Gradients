class ExplainerBase(object):

    def __init__(self, model_interface, data_interface):

        self.model_interface = model_interface
        self.data_interface = data_interface

    def generate_counterfactuals(self):

        raise NotImplementedError

    def generate_nearest_CF_neighbour(self):

        raise NotImplementedError
    
    def plot_explanation(self):
        
        raise NotImplementedError
