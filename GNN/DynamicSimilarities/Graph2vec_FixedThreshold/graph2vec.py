import os
import pickle
import random
import numpy as np
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

'''
def load_graphs(pkl_filepath, num_graphs=7):
    """
    Carrega os grafos do ficheiro pickle e garante que possuem atributos de nó.
    """
    if not os.path.exists(pkl_filepath):
        raise FileNotFoundError(f"O ficheiro não existe: {pkl_filepath}")
 
    with open(pkl_filepath, 'rb') as f:
        all_graphs = pickle.load(f)
 
    print(f"Sucesso: {len(all_graphs)} grafos carregados.")
    sample_graphs = all_graphs[:num_graphs]
   
    # Garantir que cada nó tem um atributo 'feature' para o algoritmo WL
    # Se o teu grafo já tiver condições/labels, o código abaixo deve ser ajustado
    # para usar esses nomes de atributos existentes.
    for G in sample_graphs:
        for node in G.nodes():
            if 'feature' not in G.nodes[node]:
                # Se não houver atributo, usamos o grau como condição repetível
                G.nodes[node]['feature'] = str(G.degree(node))
               
    return sample_graphs
'''
class CustomGraph2Vec:
    def __init__(self, dimensions=64, wl_iterations=2, epochs=50, workers=1, min_count=1, seed=42):
        self.dimensions = dimensions
        self.wl_iterations = wl_iterations
        self.epochs = epochs
        self.workers = workers
        self.min_count = min_count
        self.seed = seed
        self.model = None
        self.embeddings = None

        # Bloquear sementes globalmente para garantir determinismo
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    def fit(self, graphs):
        unique_tagged_data = []
        signature_to_tag = {}  # Dicionário que mapeia a estrutura única para uma tag ("tipo_0", "tipo_1", ...)
        self.graph_tags = []   # Guarda a tag de cada grafo (preservando a ordem original)
       
        print("A extrair subestruturas via WL e a isolar grafos únicos para treino...")
        for i, G in enumerate(graphs):
            # Garantir que os grafos trazem sempre a característica `feature` associada ao Node ID (ou Degree) para evitar KeyErrors
            for node in G.nodes():
                # We also check if the dictionary itself is completely empty to be safe
                if 'feature' not in G.nodes[node] or G.nodes[node]['feature'] is None:
                    # SECRETO: Se usarmos G.degree(node), todos os grafos estrela com 6 nós são 100% IDÊNTICOS para o modelo!
                    # Para o modelo saber distinguir "QUEM" são os nós que entraram/saíram, a feature tem de ser o ID do Produto!
                    G.nodes[node]['feature'] = str(node)
            
            # 1. Verificar se o grafo está vazio
            if G.number_of_nodes() == 0:
                # Se vazio, criamos uma palavra genérica para não quebrar o treino
                substructures = ["empty_graph"]
            else:
                try:
                    # Tenta extrair hashes dos nós (Requer NX 2.8+)
                    node_hashes_dict = nx.weisfeiler_lehman_node_hashes(G, node_attr='feature', iterations=self.wl_iterations)
                    
                    # Ordenamos forçadamente as iterações e inserimos os valores respeitando a ordem lexical/id de cada nó.
                    substructures = []
                    for node in sorted(G.nodes()):
                        if node in node_hashes_dict:
                            substructures.extend(node_hashes_dict[node])
                except AttributeError:
                    # Fallback para versões antigas
                    substructures = []
                    for it in range(1, self.wl_iterations + 1):
                        substructures.append(nx.weisfeiler_lehman_graph_hash(G, node_attr='feature', iterations=it))
                        
                # Ordenar as substructures para forçar simetria nos embeds em grafos topologicamente idênticos!
                substructures = sorted(substructures)
           
            # 2. Garantir que a lista nunca é vazia
            if not substructures:
                substructures = [f"node_count_{G.number_of_nodes()}"]
            
            # --- Identificar Grafos Únicos sem usar Hashes (via tuplo estrutural/assinatura legível) ---
            # Usamos o tuplo nativo do Python (pois os arrays são mutáveis, mas os tuplos podem ser chaves de dicionário)
            g_signature = tuple(substructures)
            
            # Treinamos APENAS se o grafo for efetivamente novo/diferente
            if g_signature not in signature_to_tag:
                # Criar um ID legível e sequencial para este tipo de grafo (exemplo: "tipo_0", "tipo_1")
                new_tag = f"tipo_{len(signature_to_tag)}"
                signature_to_tag[g_signature] = new_tag
                
                # A tag "tipo_X" é atribuída a este novo dicionário de "Tipos de Grafo".
                unique_tagged_data.append(TaggedDocument(words=substructures, tags=[new_tag]))
            
            # Registamos a tag na ordem cronológica à qual este grafo #i sempre correspondeu
            current_tag = signature_to_tag[g_signature]
            self.graph_tags.append(current_tag)
     
        # 3. Configuração do Doc2Vec para evitar o erro de 'high <= 0'
        # dm=0 seleciona PV-DBOW, que é mais rápido e não exige janela (window) de contexto
        
        # Para OBRIGAR a inicialização determinística da rede neural no Gensim independentemente do sistema:
        import hashlib
        def deterministic_hash(text):
            return int(hashlib.md5(str(text).encode('utf-8')).hexdigest(), 16) & 0xffffffff
            
        self.model = Doc2Vec(
            vector_size=self.dimensions,
            dm=0,                             # 0 = PV-DBOW (melhor para Graph2Vec)
            min_count=self.min_count,         # Não descarta nenhuma subestrutura rara
            workers=1,                        # ATENÇÃO: workers > 1 quebra OBJETIVAMENTE o determinismo no Gensim (threads no numpy)!
            seed=self.seed,
            epochs=self.epochs,
            hashfxn=deterministic_hash        # A hash nativa do Python muda a cada run. Esta não muda.
        )
       
        print(f"A construir vocabulário para {len(unique_tagged_data)} assinaturas únicas de grafo...")
        self.model.build_vocab(unique_tagged_data)
       
        print(f"A treinar em {len(unique_tagged_data)} grafos (ignorando redundâncias topológicas)...")
        self.model.train(unique_tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
     
        # 4. Emitir Embeddings Inferidos / Questionar o Modelo Treinado 
        # (Para cada grafo original iterado, pedimos o Embedding referente à sua tag correspondente)
        self.embeddings = np.array([self.model.dv[tag] for tag in self.graph_tags])

    def get_embedding(self):
        return self.embeddings
        
    def infer(self, graphs):
        """
        Infer embeddings for new/unseen graphs using the trained model.
        """
        if self.model is None:
            raise ValueError("You must fit the model before calling infer().")
            
        inferred_embeddings = []
        for G in graphs:
            for node in G.nodes():
                if 'feature' not in G.nodes[node] or G.nodes[node]['feature'] is None:
                    G.nodes[node]['feature'] = str(node)
                    
            if G.number_of_nodes() == 0:
                substructures = ["empty_graph"]
            else:
                try:
                    node_hashes_dict = nx.weisfeiler_lehman_node_hashes(G, node_attr='feature', iterations=self.wl_iterations)
                    substructures = []
                    for node in sorted(G.nodes()):
                        if node in node_hashes_dict:
                            substructures.extend(node_hashes_dict[node])
                except AttributeError:
                    substructures = []
                    for it in range(1, self.wl_iterations + 1):
                        substructures.append(nx.weisfeiler_lehman_graph_hash(G, node_attr='feature', iterations=it))
                        
                substructures = sorted(substructures)
                
            if not substructures:
                substructures = [f"node_count_{G.number_of_nodes()}"]
                
            # Use Doc2Vec's infer_vector to get embedding for this new document
            inferred_vector = self.model.infer_vector(substructures, epochs=20, alpha=0.025)
            inferred_embeddings.append(inferred_vector)
            
        return np.array(inferred_embeddings)
 
if __name__ == '__main__':
    # Configuração de caminhos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    distance= 'cid'
    window_size = 15
    step_size = 1
    percentile = 0.5
    product_id = 907969
    enable_edges_within_star = False
    prefix = "" if enable_edges_within_star else "star_"
    pkl_path = os.path.join(
        current_dir,
        '..', 'GraphAnalysis', 'DynamicGraphPkls', distance, str(window_size), str(step_size), str(product_id),
        f'{prefix}dynamic_graphs_{distance}_Window{window_size}_Step{step_size}_pct{percentile}.pkl'
    )
 
    try:
        # 1. Carregar
        with open(pkl_path, 'rb') as f:
            graphs_data = pickle.load(f)
            
        print(f"Sucesso: {len(graphs_data)} grafos carregados diretamente.")
       
        print("\n--- Verificação das Matrizes de Adjacência ---")
        for i, idx in enumerate(range(min(6, len(graphs_data)))): # Imprimir apenas para os primeiros 6 grafos
            G = graphs_data[idx]
            nodes = sorted(list(G.nodes()))
            adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
            print(f"Grafo {i} - Nós (ordenados): {nodes}")
            print(f"Matriz de Adjacência (pesos das arestas):\n{adj_matrix}\n")

        # 2. Gerar Embeddings
        # Aumentei as epochs para 50 para melhor convergência em amostras pequenas
        model = CustomGraph2Vec(dimensions=32, epochs=100, seed=42)
        model.fit(graphs_data)
        vectors = model.get_embedding()
       
        # 3. Resultados
        print("\n--- Resultados ---")
        print(f"Matriz de embeddings: {vectors.shape}")
        print(f"Vetor do Grafo 0 (primeiros 5 elementos): {vectors[0][:5]}")
        print(f"Vetor do Grafo 1 (primeiros 5 elementos): {vectors[1][:5]}")

        # 4. Guardar Embeddings
        import pandas as pd
        
        output_dir = os.path.join(current_dir, 'Embeddings', distance, str(window_size), str(step_size), str(product_id))
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, f'{prefix}embeddings_{distance}_Window{window_size}_Step{step_size}_pct{percentile}.csv')
        out_pkl_path = os.path.join(output_dir, f'{prefix}embeddings_{distance}_Window{window_size}_Step{step_size}_pct{percentile}.pkl')
        
        # Salvar em CSV
        df_emb = pd.DataFrame(vectors)
        df_emb.to_csv(csv_path, index=False)
        
        # Salvar em PKL
        with open(out_pkl_path, 'wb') as f:
            pickle.dump(vectors, f)
            
        print(f"\nEmbeddings guardadas com sucesso em:")
        print(f" CSV: {csv_path}")
        print(f" PKL: {out_pkl_path}")
       
    except FileNotFoundError as e:
        print(f"Erro: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
 
 