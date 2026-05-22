"""
GAT + Linear  (GATEncoder + nn.Linear)
Single ego-graph per sample.  No sequential processing.
"""
import os, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_IN='#AED6F1'; C_FEAT='#FDEBD0'; C_GRAPH='#A9DFBF'; C_GAT='#D7BDE2'
C_HEAD='#F1948A'; ARROW='#1A252F'; EDGE='#2C3E50'

def draw():
    fig, ax = plt.subplots(figsize=(20,11))
    ax.set_xlim(0,20); ax.set_ylim(0,11); ax.axis('off')
    fig.patch.set_facecolor('#FAFAFA')

    def box(x,y,w,h,color,lines,fs=8.5,bold_first=False):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.12",
            facecolor=color,edgecolor=EDGE,linewidth=1.6,zorder=3))
        if isinstance(lines,str): lines=[lines]
        n=len(lines)
        for i,l in enumerate(lines):
            dy=(i-(n-1)/2)*(fs+1.5)*0.014
            ax.text(x,y-dy,l,ha='center',va='center',fontsize=fs,
                fontweight='bold' if (bold_first and i==0) else 'normal',zorder=4,color='#1A252F')

    def arr(x1,y1,x2,y2,lbl='',sty='arc3,rad=0.0'):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
            arrowprops=dict(arrowstyle='-|>',color=ARROW,lw=1.6,
                connectionstyle=sty,mutation_scale=14),zorder=5)
        if lbl:
            ax.text((x1+x2)/2+0.05,(y1+y2)/2+0.1,lbl,fontsize=7,color='#555',zorder=6,style='italic')

    XA,XB,XC,XD,XE,XF=1.7,4.7,8.0,11.5,15.2,18.5
    for bx,bw,lbl,col in [
        (XA,2.6,'Data Inputs','#EBF5FB'),(XB,2.7,'Node Features','#FEF9E7'),
        (XC,3.0,'Ego-Graph (x1)','#EAFAF1'),(XD,3.0,'GAT Encoder','#EAFAF1'),
        (XE,3.0,'Central Node z','#FEF9E7'),(XF,2.5,'Linear Head / Output','#F5EEF8')]:
        ax.add_patch(FancyBboxPatch((bx-bw/2,0.3),bw,10.0,boxstyle="round,pad=0.1",
            facecolor=col,edgecolor='#BFC9CA',linewidth=1.0,zorder=1,alpha=0.5))
        ax.text(bx,10.5,lbl,ha='center',va='center',fontsize=8.5,color='#5D6D7E',fontweight='bold',zorder=4)

    # A
    box(XA,9.0,2.4,0.9,'#D5DBDB',['df_wide','(items x dates)'],fs=8,bold_first=True)
    box(XA,7.2,2.4,0.9,C_IN,['Historical Sales (scaled)','lookback window'],fs=8,bold_first=True)
    box(XA,5.5,2.4,0.9,C_IN,['Calendar Features','cal_dim = 31 cols'],fs=8,bold_first=True)

    # B
    box(XB,8.2,2.6,2.0,C_FEAT,['TARGET NODE','-----','ts_lookback (dims)','cal_next_step (31 dims)',
        'stats_8: last, mean7, mean,','  std, zero_ratio, slope,','  min, max','feature_dim = total'],fs=7.5,bold_first=True)
    box(XB,5.5,2.6,1.4,C_FEAT,['NEIGHBOR NODES','-----','ts_window right-aligned',
        'cal=0  (unknown)','+ stats_8'],fs=7.5,bold_first=True)
    arr(XA+1.2,7.2,XB-1.3,8.3,lbl='lookback ts')
    arr(XA+1.2,5.5,XB-1.3,7.5,lbl='cal next')
    arr(XA+1.2,9.0,XB-1.3,5.5,lbl='window data')

    # C
    box(XC,7.2,2.9,3.0,C_GRAPH,['Ego-Graph  (ONE per sample)','---------------------------',
        'Pairwise similarity (Spearman)','Threshold -> edges',
        'Star topology + within-star links','target_id -> node index 0',
        'edge_attr = similarity weight','PyG Data(x, edge_index, edge_attr)'],fs=7.5,bold_first=True)
    ax.text(XC,10.5,'edge_dim=1  (similarity weights)',ha='center',fontsize=7.5,color='#1A5276',fontweight='bold',zorder=6)
    arr(XB+1.35,8.2,XC-1.45,7.8); arr(XB+1.35,5.5,XC-1.45,6.5)

    # D – GAT
    box(XD,9.2,2.9,0.9,C_GAT,['GATConv 1  (multi-head attention)','in_ch -> hidden_ch, heads=4, concat=True',
        'edge_dim=1 (edge weights)  =>  out: hidden*4'],fs=7.4,bold_first=True)
    box(XD,7.8,2.9,0.7,C_GAT,['ELU + Dropout(p=0.2)'],fs=8)
    box(XD,6.8,2.9,0.9,C_GAT,['GATConv 2  (single-head)','hidden*4 -> out_ch, heads=1, concat=False',
        'edge_dim=1  =>  out: out_ch'],fs=7.4,bold_first=True)
    box(XD,5.3,2.9,0.85,C_GAT,['Node Embeddings','(total_nodes, out_ch)'],fs=8,bold_first=True)
    ax.annotate('',xy=(XD-1.45,9.2),xytext=(XC+1.45,7.2),
        arrowprops=dict(arrowstyle='-|>',color=ARROW,lw=1.4,connectionstyle='arc3,rad=-0.25'),zorder=5)
    arr(XD,8.75,XD,8.15); arr(XD,7.45,XD,7.25); arr(XD,6.35,XD,5.72)

    # E
    box(XE,5.3,2.8,1.0,C_GAT,['Extract Central Node','idx = ptr[:-1]  (node 0 per graph)',
        'z_target : (B, out_ch)'],fs=7.8,bold_first=True)
    arr(XD+1.45,5.3,XE-1.4,5.3)
    ax.text(XE,9.5,'No recurrence / no temporal loop',ha='center',fontsize=8,color='#1A5276',style='italic',fontweight='bold',zorder=6)
    ax.text(XE,8.9,'Single embedding per sample',ha='center',fontsize=7.5,color='#555',style='italic',zorder=6)

    # F
    box(XF,6.8,2.4,0.9,C_HEAD,['Linear(out_ch -> horizon)','horizon = 1  (next step)'],fs=7.8,bold_first=True)
    box(XF,5.2,2.4,0.9,'#A9DFBF',['Reshape','(B, horizon, 1)'],fs=8,bold_first=True)
    box(XF,3.5,2.4,0.85,'#A9DFBF',['Predictions','(B, horizon, 1)'],fs=8.5,bold_first=True)
    arr(XE+1.4,5.3,XF-1.2,6.8,lbl='z')
    arr(XF,6.35,XF,5.65); arr(XF,4.75,XF,3.93)

    ax.add_patch(FancyBboxPatch((XE-1.4,0.3),(XF+1.2-XE+1.4),1.4,boxstyle="round,pad=0.08",
        facecolor='#FDFEFE',edgecolor='#884EA0',linewidth=1.3,linestyle=':',zorder=2))
    ax.text((XE+XF)/2,1.05,
        'Recursive Inference: forecast(t) -> appended to ts_lookback -> ego-graph rebuilt -> step t+1\n'
        '(152 autoregressive steps for full horizon)',
        ha='center',va='center',fontsize=7.2,color='#6C3483',zorder=6)

    ax.text(10,10.5,'GAT + Linear Forecaster (GATEncoder) -- Architecture Block Diagram',
        ha='center',va='center',fontsize=12,fontweight='bold',color='#1A252F',zorder=6)

    plt.tight_layout(pad=0.4)
    out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'architecture_gat_linear.png')
    plt.savefig(out,dpi=160,bbox_inches='tight',facecolor=fig.get_facecolor())
    print(f'Saved -> {out}')
    plt.show()

if __name__=='__main__':
    draw()
