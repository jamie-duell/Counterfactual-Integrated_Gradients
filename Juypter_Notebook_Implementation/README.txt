For implementation using the ipynb, save in an accessible directory and use the following: 
%run CF_IG_Simple_PyTorch.ipynb
CF_IG = CounterfactualIG(model, data) 

generate an explanation array with the following: 
explanation = CF_IG.generate_counterfactuals(query_instance=[input_tensor[idx:idx+1]], counterfactual=[counterfactual[idx]])

plot the explanation with the following: 
CF_IG.plot_explanation(explanation[0], input_tensor[idx:idx+1][0], counterfactual[0])

