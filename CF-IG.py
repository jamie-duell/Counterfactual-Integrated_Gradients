from Explainer_SetUp import ExplainerBase
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad
from scipy.spatial import distance
import matplotlib.pyplot as plt


class CounterfactualIG(ExplainerBase):

    def __init__(self, model_interface, data_interface):

        super().__init__(model_interface, data_interface)
    
    
    def generate_nearest_CF_neighbour(self, query_instance):
        #cf_pred = self.model_interface(query_instance)
        #target = np.argmin(cf_pred.detach().numpy())
        #dist = distance.squareform(distance.pdist(self.data_interface)) #distance between each data points w.r.t all other data points
        #nearest_neighbours = np.argsort(dist, axis=1)  #number of closest points
        #nearest_neighbours_arr = nearest_neighbours[inp_idx]
        #for i in range(len(nearest_neighbours_arr)-1):
        #    if np.round(self.model_interface(self.data_interface[nearest_neighbours_arr[i]])[target].detach().numpy()) != np.round(self.model_interface(self.data_interface[inp_idx])[target].detach().numpy()): #convert to numpy from tensor
        #        closest_instance = data[nearest_neighbours_arr[i]]  
        
        
        return print("This function is not yet implemented in the non-client side version of this method used in the paper \n simple adaptation of the code in the source file are necessary to make this work in the general case.")
        
    
    def generate_counterfactuals(self, query_instance, counterfactual, target = 0.5, _K=500, decision_boundary_proba=0.5):
       
        """ 
        
        Requires both query_instance and counterfactual to be tensors in an array - working to fix this:
        
        Currently, the input must be: [input_tensor[a:a+1]] and [counterfactual_tensor] associated with [input_tensor[a:a+1]]
        
        """
        self._K = _K #number of steps in the Riemann Approximation
        #query_instance = torch.FloatTensor(query_instance) 
        #counterfactual = torch.FloatTensor(counterfactual)

        # k/m in the formula
        alphas = torch.linspace(0, 1, _K).tolist()
    
        # direct path from baseline to input. shape : ([n_steps, n_features], )
        scaled_features = tuple(
                torch.cat(
                    [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
                ).requires_grad_()
                for input, baseline in zip(counterfactual, query_instance)
            )
    
        # predictions at every step. shape : [n_steps, 1]
        cf_pred = self.model_interface(counterfactual[0])
        target = np.argmax(cf_pred.detach().numpy()) #index returned by arg max of the greatest pred probability aka. if 0 is highest then pred = 0
        preds = self.model_interface(scaled_features[0])[:, target] 
    
        cf_exp = scaled_features
    
        index = min(find_indices(self.model_interface(cf_exp[0])[:,target], lambda e: e >= decision_boundary_proba))
    
        new_feature = scaled_features[0][index]
    
        new_scaled_features = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(new_feature, query_instance)
        )
    
        preds2 = self.model_interface(new_scaled_features[0])[:,target]
    
        # gradients of predictions wrt input features. shape : [n_steps, n_features]
    
        #grads = grad(outputs=torch.unbind(preds), inputs=scaled_features)
        grads = grad(outputs=torch.unbind(preds2), inputs=new_scaled_features)
    
        explanation_a = grads[0].mean(0) #get the mean gradient between both points
        
        explanation = explanation_a*(new_feature[0].detach().numpy() - query_instance[0].detach().numpy())
            
        return explanation
    
    def find_indices(lst, condition):  #used to find the index of the minimum value over decision bounds for target
        return [i for i, elem in enumerate(lst) if condition(elem)] 

    
    def plot_explanation(self, explanation_array, query_instance, counterfactual):
        """ 
        
        explanation_array = explanation provided by generate_counterfactuals()
        
        query_instance = use the same instance as input into generate_counterfactuals() but as a tensor,
        thus written as input_tensor[a:a+1] not [input_tensor[a:a+1]]
        
        counterfactual_instance = the same counterfactual input into generate_counterfactuals() but as a tensor,
        thus written as counterfactual_tensor not [counterfactual_tensor]
        
        """
        explanation = explanation_array.detach().numpy()
        increase_or_decrease_for_counterfactual = query_instance.detach().numpy() - counterfactual.detach().numpy()
        feature_names = self.data_interface.columns.tolist()
        
        
        result_increase_cf = [np.sign(val) for val in increase_or_decrease_for_counterfactual]
        third_array_values = ['Increased Value' if val < 0 else 'Decreased Value' if val > 0 else 'No Change' for val in result_increase_cf]
        
        
        font = {'family':'normal',
               'weight':'bold',
               'size': 12}
        plt.rc('font', **font)
        # Append each value from third_array_values to the corresponding feature name
        # Format: feature_name (value)
        for idx, value in enumerate(third_array_values):
            feature_names[idx] += f' ({value})'

        # Create a figure and axis
        %matplotlib qt


        fig, ax = plt.subplots(figsize=(10, 6))
        bars =[]
        # Plot the horizontal bars
        bars = []
        for idx, value in enumerate(explanation):
            color = 'red' if value < 0 else 'green'
            bar = ax.barh(feature_names[idx], value, color=color)
            bars.append(bar)

        # Attach data as annotations to each bar
            mplcursors.cursor(bar).connect(
                "add", lambda sel, idx=idx: sel.annotation.set_text(
                    f'Old Feature Value: {input_tensor[2:3].detach().numpy()[0][idx]}, New Feature Value: {nearest_neighbour_list[2].detach().numpy()[idx]}'
                )
            )
    
        ax.set_xlabel('Attribution Values')    
    
        #def on_hover(sel):
        #    bar_index = sel.artist.get_label()
        #    val1 = input_tensor[2:3].detach().numpy()[0][bar_index]
        #    val2 = nearest_neighbour_list[2].detach().numpy()[bar_index]
        #    sel.annotation.set_text(f'Old Feature Value: {val1}\n New Feature Value: {val2}')
        #    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

        # Add the interactivity using mplcursors
        cursor = mplcursors.cursor(bars, hover=True)
        #cursor.connect("add", on_hover)
        # Set the x-axis label
        plt.title("Counterfactual Feature Attribution")

        # Show the plot
        plt.tight_layout()
        
        return plt.show()
