import pandas as pd
import numpy as np

def filter_singleton_classes(df_pam50, df_meth, df_atac=None, min_class_size=2):
    """
    Filters out classes from df_pam50 with fewer than min_class_size samples,
    and consistently drops the corresponding patient barcodes from df_meth and df_atac.
    
    Args:
        df_pam50: DataFrame with index=patient_barcode, column 'subtype'
        df_meth: DataFrame with index=patient_barcode, columns=CpG probe IDs
        df_atac: Optional DataFrame with index=patient_barcode (or probe_id), columns=features/barcodes
        min_class_size: Minimum number of samples required for a class to be retained (default: 2)
        
    Returns:
        tuple: (df_pam50_filtered, df_meth_filtered, df_atac_filtered, dropped_info)
    """
    class_counts = df_pam50['subtype'].value_counts()
    small_classes = class_counts[class_counts < min_class_size].index.tolist()
    
    dropped_info = {}
    if small_classes:
        mask_drop = df_pam50['subtype'].isin(small_classes)
        dropped_barcodes = df_pam50[mask_drop].index.tolist()
        
        for sc in small_classes:
            dropped_info[sc] = {
                "count": int(class_counts[sc]),
                "barcodes": df_pam50[df_pam50['subtype'] == sc].index.tolist()
            }
        
        dropped_str = ', '.join(f"{k} (n={v['count']})" for k, v in dropped_info.items())
        print(f"NOTE: Filtering out {len(small_classes)} class(es) with < {min_class_size} samples: {dropped_str}")
        print(f"Dropped {len(dropped_barcodes)} patient(s): {dropped_barcodes}")
        
        df_pam50 = df_pam50[~mask_drop]
        df_meth = df_meth.loc[df_pam50.index]
        
        if df_atac is not None:
            if set(df_pam50.index).issubset(set(df_atac.index)):
                df_atac = df_atac.loc[df_pam50.index]
            elif set(df_pam50.index).issubset(set(df_atac.columns)):
                df_atac = df_atac[df_pam50.index]
            else:
                common_barcodes = [b for b in df_pam50.index if b in df_atac.index]
                if common_barcodes:
                    df_atac = df_atac.loc[common_barcodes]
                elif [b for b in df_pam50.index if b in df_atac.columns]:
                    df_atac = df_atac[[b for b in df_pam50.index if b in df_atac.columns]]
            
        print(f"Cohort size adjusted: {len(mask_drop)} -> {len(df_pam50)} patients.")
    else:
        print(f"All classes meet min_class_size={min_class_size}. No filtering needed.")
        
    return df_pam50, df_meth, df_atac, dropped_info

if __name__ == "__main__":
    barcodes = [f"TCGA-A2-{i:04d}" for i in range(43)]
    subtypes = ['LumA']*18 + ['LumB']*11 + ['Basal']*10 + ['Normal']*3 + ['HER2']*1
    df_p = pd.DataFrame({'subtype': subtypes}, index=barcodes)
    df_m = pd.DataFrame(np.random.rand(43, 10), index=barcodes)
    df_a = pd.DataFrame(np.random.rand(43, 5), index=barcodes)
    
    df_p_filt, df_m_filt, df_a_filt, dropped = filter_singleton_classes(df_p, df_m, df_a, min_class_size=2)
    assert len(df_p_filt) == 42
    assert len(df_m_filt) == 42
    assert len(df_a_filt) == 42
    assert 'HER2' not in df_p_filt['subtype'].values
    print("Self-test passed: HER2 (n=1) successfully filtered from all modalities.")
