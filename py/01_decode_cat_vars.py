def decode_cat_var(data, variables, target):
    
    """
    Decode categorical variables.
    Replace variables with the corresponding Weight of Evidence value and store variable by 'DEC_' prefix.
    """
    
    group_table_all = pd.DataFrame()
    
    for var in variables:
        
        group_table = pd.DataFrame()
        group_table['cnt'] = data[var].value_counts(dropna=False)
        group_table['bad'] = data.groupby(var)[target].sum().replace({0: 0.00001})
        group_table['good'] = (group_table['cnt'] - data.groupby(var)[target].sum().replace({0: 0.00001})).replace({0: 0.00001})
        group_table['pct_bad'] = group_table['bad'] / group_table['bad'].sum()
        group_table['pct_good'] = group_table['good'] / group_table['good'].sum()
        group_table['woe'] = np.log(group_table['pct_good']/group_table['pct_bad'])
                
        data['DEC_'+var] = data[var]
        
        data = data.replace({'DEC_'+var: group_table.to_dict('dict')['woe']})
    
        group_table['variable_name'] = var
        group_table = group_table.reset_index().rename(columns={'index': 'value'})
        group_table = group_table[['variable_name','value','woe']]
        
        group_table_all = group_table_all.append(group_table)
        
    return data, group_table_all


def transfer_cat_var(data, variables, cat_table):
    
    """
    Transfer categorical variables to test sample.
    Replace variables with the corresponding Weight of Evidence value from train sample and store variable by 'DEC_' prefix.
    """
        
    for var in variables:
            
        data['DEC_'+var] = data[var]
        data = data.replace({'DEC_'+var: cat_table[cat_table['variable_name'] == var].set_index('value').to_dict('dict')['woe']})
            
        data['DEC_'+var] = np.where(data['DEC_'+var].isin(list(cat_table.loc[cat_table['variable_name'] == var]['woe'])), data['DEC_'+var], np.nan)
            
    return data


train_data, cat_table = decode_cat_var(train_data, features_cat, 'defaulted_fl')
test_data = transfer_cat_var(test_data, features_cat, cat_table)