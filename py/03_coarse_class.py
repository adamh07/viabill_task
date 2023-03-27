from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
import math

def coarse_class(data, variables, target, groups):
    
    """
    Merge adjacent bins of fine class procedure based on the preference of group number.
    """
    
    def monotonic(x):
            
        dx = np.diff(x)
        mn = np.all(dx <= 0) or np.all(dx >= 0)
        if mn == True:
            return 1
        else:
            return 0
    
    stats_all = pd.DataFrame()
    ivgini_all = pd.DataFrame()
    
    for var in variables:
        
        stats_var = pd.DataFrame()
        ivgini_var = pd.DataFrame()        
        
        for i in range(2,groups+1):
            
            dtree = DecisionTreeClassifier(criterion='gini',min_impurity_decrease=0, max_leaf_nodes=i,
                                           min_samples_leaf=round(math.ceil(data.shape[0]*0.05)))
            
            d = dtree.fit(np.array(data.loc[data['GRP_'+var] != -1]['GRP_'+var]).reshape(-1,1),
                          data.loc[data['GRP_'+var] != -1][target])
            
            thresholds = d.tree_.threshold
            
            l = [0]
            for t in thresholds:
                if t != _tree.TREE_UNDEFINED:
                    l.append(t)
            l.append(len(data.loc[data['GRP_'+var] != -1]['GRP_'+var].unique()))
            l.sort()
            
            c = pd.cut(np.array(data['GRP_'+var]), bins=l, right=False)
            if (i > 2) & (len(c.value_counts()) < i): 
                
                break
                
            if c.value_counts().count() == 1:
                
                c = pd.cut(np.array(data['GRP_'+var]), bins=data['GRP_'+var].value_counts().count(), right=False)
            
            c.categories = range(0,len(c.value_counts()))
            data["C"+str(i)+"_"+var] = c
            
            if data[var].isna().sum() != 0:
                
                data["C"+str(i)+"_"+var] = data["C"+str(i)+"_"+var].cat.add_categories(-1)
                data["C"+str(i)+"_"+var] = data["C"+str(i)+"_"+var].fillna(-1)
                
            stats, ivgini = woe_calc(data,var,"C"+str(i)+"_"+var,target) 
            ivgini['monotonous_woe'] = monotonic(stats[(stats['grp_name'] == "C"+str(i)+"_"+var) & (stats['grp'] != -1)]['woe'])
            
            stats_var = stats_var.append(stats)
            ivgini_var = ivgini_var.append(ivgini)
            
        stats_all = stats_all.append(stats_var)
        ivgini_all = ivgini_all.append(ivgini_var)
        
    return data, stats_all, ivgini_all


train_data, coarse_stats, coarse_ivgini = coarse_class(train_data, features_fine, 'defaulted_fl', 5)