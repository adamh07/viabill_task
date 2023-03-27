def corr_plot(data, variables, method):
    
    
    corrdata = data[variables].astype(float)    
    
    corr = corrdata.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    np.fill_diagonal(mask, False)
    
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    
    cm = sns.heatmap(corr, xticklabels=variables, yticklabels=variables, mask=mask, 
            cmap='Greys', center=0, annot=True, annot_kws={"size": 10}, cbar_kws={"shrink": 1})
    plt.title(str(method.capitalize()) + ' correlation matrix', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    fig.patch.set_facecolor('white')
    ax.set_facecolor('whitesmoke')
    
    for _, spine in cm.spines.items():
        spine.set_visible(True)

    plt.show()
    
    
def filter_by_corr(data, variables, method, threshold=0.5):
        
    matrix = data[variables].corr(method = method)
        
    lower = matrix.abs().where(np.tril(np.ones(matrix.shape),k=-1).astype(np.bool))
    corr_cols = lower.columns

    for col in corr_cols:
    
        drop_list = []
    
        if col in lower.columns:
        
            for row in range(0, lower.shape[0]):
        
                if (lower[col].iloc[row] > threshold) & (np.isnan(lower[col].iloc[row]) == False):
                
                    drop_list.append(lower[col].index[row])
                    corr_cols.drop(lower[col].index[row])
                
            lower = lower.drop(columns = drop_list)
            lower = lower.drop(drop_list)

    return(list(lower.columns))