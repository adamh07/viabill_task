def woe_calc(data, orig_var, grp_var, target):

    """
    Create a summary table of groups and give the predictor power of the variable as information value and gini.
    """    
    
    group_table = pd.DataFrame()
    group_table['orig_name'] = ()
    group_table['grp_name'] = ()
    group_table['grp'] = ()    
    group_table['cnt'] = data[grp_var].value_counts(dropna=False)
    group_table['bad'] = data.groupby(grp_var)[target].sum().replace({0: 0.00001}) 
    group_table['good'] = (group_table['cnt'] - data.groupby(grp_var)[target].sum()).replace({0: 0.00001})
    group_table['min'] = data.groupby(grp_var)[orig_var].min()
    group_table['max'] = data.groupby(grp_var)[orig_var].max()
    group_table['pct_total'] = group_table['cnt'] / group_table['cnt'].sum()
    group_table['pct_bad'] = group_table['bad'] / group_table['bad'].sum()
    group_table['pct_good'] = group_table['good'] / group_table['good'].sum()
    group_table['target_rate'] = group_table['bad'] / group_table['cnt']
    group_table['woe'] = np.log(group_table['pct_good']/group_table['pct_bad'])
    group_table['iv'] = (group_table['pct_good'] - group_table['pct_bad']) * group_table['woe']
    group_table['orig_name'] = orig_var
    group_table['grp_name'] = grp_var
    group_table['grp'] = group_table.index
    group_table['grp'] = group_table['grp'].astype(np.int64)
    
    group_table = group_table.sort_values(by=['min']).reset_index(drop=True)    
        
    ivgini_table = pd.DataFrame({"variable_name": grp_var,
                                     "iv": [group_table[group_table['grp_name'] == grp_var]['iv'].sum()],
                                     "gini": [abs(roc_auc_score(data[target],data[grp_var])*2-1)]})  
    
    return group_table, ivgini_table




def fine_class(data, variables, target, groups, discrete_limit):

    """
    Create equally-sized bins based on the preference of group number.
    If the number of unique values does not reach the discrete limit parameter, the fine groups will be the raw values.
    """    
    
    stats_all = pd.DataFrame()
    ivgini_all = pd.DataFrame()
    
    for var in variables:
        
        if isinstance(data.loc[data[var].isna() == False].reset_index()[var][0], str):
            
            var = 'DEC_'+var
            
        if data[var].value_counts().count() <= discrete_limit:
            
            u = data[var].unique()
            u.sort()
            l = [-np.inf]+list(u)
            
            data['GRP_'+var] = pd.cut(data[var],l,labels=False).astype('category')
            
        else:
            
            data['GRP_'+var] = pd.qcut(data[var],groups,duplicates='drop',labels=False).astype('category')
            
        if data[var].isna().sum() != 0:
            
            data['GRP_'+var] = data['GRP_'+var].cat.add_categories(-1)
            data['GRP_'+var] = data['GRP_'+var].fillna(-1)
            
        stats, ivgini = woe_calc(data, var, 'GRP_'+var, target)
        stats_all = stats_all.append(stats)
        ivgini_all = ivgini_all.append(ivgini)
            
    return data, stats_all, ivgini_all


train_data, fine_stats, fine_ivgini = fine_class(train_data, features, 'defaulted_fl', 20, 20)