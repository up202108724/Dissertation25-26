import os
import pandas as pd
import plotly.graph_objects as go

def visualize_product_ts(df_wide, item_ids, train_size, val_size, output_dir="TSPLots"):
    """
    Plots the train and validation sets for specific item_ids and saves them.
    """
    # Criar a pasta principal se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Extrair as colunas de data do DataFrame Pivotado
    all_dates = df_wide.columns
    train_dates = all_dates[:train_size]
    val_dates = all_dates[train_size:train_size + val_size]

    for item_id in item_ids:
        if item_id not in df_wide.index:
            print(f"Item {item_id} not found in df_wide. Skipping...")
            continue

        fig = go.Figure()
        
        # Dados de Treino
        y_train = df_wide.loc[item_id, train_dates]
        fig.add_trace(go.Scatter(x=train_dates, y=y_train, mode='lines', name='Train Set', line=dict(color='blue')))
        
        # Dados de Validação
        y_val = df_wide.loc[item_id, val_dates]
        fig.add_trace(go.Scatter(x=val_dates, y=y_val, mode='lines', name='Validation Set', line=dict(color='orange')))

        # Configurações do Gráfico
        fig.update_layout(
            title=f"Time Series for Item ID: {item_id}",
            xaxis=dict(
                title="Date",
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            ),
            yaxis_title="Value (Sales/Demand)",
            template="plotly_white"
        )

        # Salvar o gráfico como HTML interativo
        item_dir = os.path.join(output_dir, str(item_id))
        if not os.path.exists(item_dir):
            os.makedirs(item_dir)
            
        file_path = os.path.join(item_dir, f"product_{item_id}_ts.html")
        fig.write_html(file_path)
        print(f"Saved plot for item {item_id} to {file_path}")

if __name__ == "__main__":
    # --- Configurações conforme o seu setup ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'data_andre.feather')
    
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_feather(DATA_PATH)
    
    # Pivotar os dados (long to wide)
    df_wide = df.pivot_table(index='item_id', columns='date', values='value', aggfunc='sum').fillna(0)
    
    # Definir tamanhos conforme o seu histórico de 2 anos (455 treino + 154 validação)
    train_size = 455
    val_size = 154
    
    # Output path for the TSPLots folder
    output_folder = os.path.join(BASE_DIR, 'TSPLots')
    
    for product_id in df['item_id'].unique():  # Visualizar os produtos
        visualize_product_ts(df_wide, [product_id], train_size, val_size, output_dir=output_folder)