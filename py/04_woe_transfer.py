def woe_changer(data, variables, stats):
    
    """
    Create Weight of Evidence-decoded variables on the train sample by the summary table of coarse classing.
    Change the group values to woe values.
    """
        
    for var in variables:
            
        data['W_'+var] = data[var]
        data = data.replace({'W_'+var: stats.loc[stats['grp_name'] == var].set_index('grp').to_dict('dict')['woe']})
        data['W_'+var] = data['W_'+var].astype(float)
        
    return data


def woe_transfer(data, variables, stats, missing_dict):
    
    """
    Transfer group and Weight of Evidence-decoded variables to test sample.
    """
    
    for var in variables:
        
        w = stats.loc[stats['grp_name'] == var]
        
        orig_name = stats.loc[stats['grp_name'] == var]['orig_name'][0]
        grp_name = stats.loc[stats['grp_name'] == var]['grp_name'][0]
        miss_cat = missing_dict[grp_name]
    
        w = w.sort_values(by=['grp']).reset_index(drop=True)
    
        l = [-np.inf]
        g = []
        
        for i in range(0,w.shape[0]):
            
            if math.isnan(w['max'][i]) == False:
                
                if i < (w.shape[0]-1):
                    
                    l.append((round(w['max'][i], 5)+round(w['min'][i+1], 5))/2)
                
                g.append(w['grp'][i])
                
        l.append(np.inf)
        
        data[grp_name] = pd.cut(data[orig_name], l, labels=g, right=True).astype('category')
        data['W_'+grp_name] = pd.cut(data[orig_name], l, labels=g, right=True).astype('category')
        
        if miss_cat == -1:
        
            data[grp_name] = data['W_'+grp_name].cat.add_categories(miss_cat)
            data['W_'+grp_name] = data['W_'+grp_name].cat.add_categories(miss_cat)
        
        data[grp_name] = data['W_'+grp_name].fillna(miss_cat)
        data['W_'+grp_name] = data['W_'+grp_name].fillna(miss_cat)
    
        data = data.replace({'W_'+grp_name: w.set_index('grp').to_dict('dict')['woe']})
        data['W_'+grp_name] = data['W_'+grp_name].astype(float)
        
    return data


train_data = woe_changer(train_data, features_coarse, coarse_stats)
test_data = woe_transfer(test_data, features_coarse, coarse_stats, miss_dict)