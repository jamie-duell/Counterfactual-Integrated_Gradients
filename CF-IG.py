from Explainer_SetUp import ExplainerBase
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad
from scipy.spatial import distance
import matplotlib.pyplot as plt


class CounterfactualIG(ExplainerBase):

    def __init__(self, model_interface, data_interface, explain_index):

        super().__init__(model_interface, data_interface, explain_index)
    
    
    def generate_nearest_CF_neighbour(self, query_instance, df_columns_to_drop, categorical_features, ohe=False):
        
        data_interface = self.data_interface
        data_interface = data_interface.drop(labels=df_columns_to_drop, axis=1)
        data_interface = data_interface.values
        data_interface = data_interface.astype(str)
        categorical_names = {}
        
        if ohe==False:
            for feature in categorical_features:
                le = LabelEncoder()
                le.fit(data_interface[:, feature])
                data_interface[:, feature] = le.transform(data_interface[:, feature])
                categorical_names[feature] = le.classes_
            encoder = OneHotEncoder().fit(data_interface) 
        else:
            pass
        
        data_interface = np.double(data_interface)
        cf_pred = self.model_interface(query_instance)
        target = np.argmin(cf_pred.detach().numpy())
        dist = distance.squareform(distance.pdist(data_interface))
        nearest_neighbours = np.argsort(dist, axis=1)
        nearest_neighbours_arr = nearest_neighbours[self.explain_index]

        for i in range(len(nearest_neighbours_arr) - 1):
            model_out_query = self.model_interface(query_instance)
            
            model_out_interface = self.model_interface(torch.from_numpy(data_interface[nearest_neighbours_arr[i]]).type(torch.FloatTensor))
            
            if np.round(model_out_interface.detach().numpy())[target] != np.round(model_out_query.detach().numpy())[target]:
                closest_point = data_interface[nearest_neighbours_arr[i]]
        return torch.from_numpy(closest_point).type(torch.FloatTensor)
        
        return print("This function is not yet implemented in the non-client side version of this method used in the paper \n simple adaptation of the code in the source file are necessary to make this work in the general case.")
        
    
    def generate_counterfactuals(self, query_instance, counterfactual, _K=5000, decision_boundary_proba=0.500000):
       
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
        not_target = np.argmax(cf_pred.detach().numpy()) 
        preds = self.model_interface(scaled_features[0]) #[:, target]
        cf_exp = scaled_features
        
        #for softmax 
        #index = max(self.find_indices(self.model_interface(cf_exp[0])[:,not_target], lambda e: e < decision_boundary_proba))
        
        #update this for < 0.5 = class 0 and >= 0.5 is class 1 
        if self.model_interface(query_instance[0]) >= decision_boundary_proba: 
            index = min(self.find_indices(self.model_interface(cf_exp[0]), lambda e: e < decision_boundary_proba))
        else:
            index = min(self.find_indices(self.model_interface(cf_exp[0]), lambda e: e > decision_boundary_proba))
    
        new_feature = scaled_features[0][index]
    
        grads = grad(outputs=torch.unbind(preds), inputs=scaled_features)
        #grads = grad(outputs=torch.unbind(preds2), inputs=new_scaled_features)
    
        explanation_a = grads[0][:index].mean(0) #get the mean gradient between both points
        
        explanation = (new_feature.detach().numpy() - query_instance[0].detach().numpy()) *explanation_a.detach().numpy()
            
        return explanation
    
    def find_indices(self, lst, condition):  #used to find the index of the minimum value over decision bounds for target
        return [i for i, elem in enumerate(lst) if condition(elem)] 

    
    def plot_explanation(self, input_tensor, explanation_array, query_instance, counterfactual):
        """ 
        
        explanation_array = explanation provided by generate_counterfactuals()
        
        query_instance = use the same instance as input into generate_counterfactuals() but as a tensor,
        thus written as input_tensor[a:a+1] not [input_tensor[a:a+1]]
        
        counterfactual_instance = the same counterfactual input into generate_counterfactuals() but as a tensor,
        thus written as counterfactual_tensor not [counterfactual_tensor]
        
        """
        #explanation = explanation_array.detach().numpy()
        increase_or_decrease_for_counterfactual = query_instance.detach().numpy() - counterfactual.detach().numpy()
        feature_names = self.data_interface.columns.tolist()
        
        cf_pred = self.model_interface(query_instance)
        target = np.argmax(cf_pred.detach().numpy())
        
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
        bars = []
        # Plot the horizontal bars
        for idx, value in enumerate(explanation):
            color = 'red' if value < 0 else 'green'
            bar = ax.barh(feature_names[idx], value, color=color)
            bars.append(bar)

        # Attach data as annotations to each bar
            mplcursors.cursor(bar).connect(
                "add", lambda sel, idx=idx: sel.annotation.set_text(
                    f'Old Feature Value: {input_tensor[self.explain_index:self.explain_index+1].detach().numpy()[idx]}, New Feature Value: {nearest_neighbour_list[self.explain_index].detach().numpy()[idx]}'
                )
            )
        cursor = mplcursors.cursor(bars, hover=True)
        true_class =  np.round(self.model_interface(query_instance).detach().numpy())[target]
        counterfactual_class = np.round(self.model_interface([counterfactual][0]).detach().numpy())[target]
        top_right_text = f'True Class: {true_class} \n Counterfactual Class: {counterfactual_class}'
        ax.text(0.35, 0.95, top_right_text, transform=ax.transAxes,
            fontsize=12, fontweight='bold', ha='right')
    
        ax.set_xlabel('Attribution Values')    
    
        #def on_hover(sel):
        #    bar_index = sel.artist.get_label()
        #    val1 = input_tensor[2:3].detach().numpy()[0][bar_index]
        #    val2 = nearest_neighbour_list[2].detach().numpy()[bar_index]
        #    sel.annotation.set_text(f'Old Feature Value: {val1}\n New Feature Value: {val2}')
        #    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

        # Add the interactivity using mplcursors
       
        #cursor.connect("add", on_hover)
        # Set the x-axis label
        plt.title("Counterfactual Feature Attribution")

        # Show the plot
        plt.tight_layout()
        
        return plt.show()
