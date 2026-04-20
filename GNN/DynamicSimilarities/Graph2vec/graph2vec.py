import os
import pickle
import random
import numpy as np
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
 
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
        tagged_data = []
       
        print("A extrair subestruturas via WL...")
        for i, G in enumerate(graphs):
            # Garantir que os grafos trazem sempre a característica `feature` associada ao Node ID (ou Degree) para evitar KeyErrors
            for node in G.nodes():
                # We also check if the dictionary itself is completely empty to be safe
                if 'feature' not in G.nodes[node] or G.nodes[node]['feature'] is None:
                    #print(f"Aviso: Nó {node} no grafo {i} não tem 'feature'. Atribuindo grau como fallback.")
                    G.nodes[node]['feature'] = str(G.degree(node))
            
            # 1. Verificar se o grafo está vazio
            if G.number_of_nodes() == 0:
                # Se vazio, criamos uma palavra genérica para não quebrar o treino
                substructures = ["empty_graph"]
            else:
                try:
                    # Tenta extrair hashes dos nós (Requer NX 2.8+)
                    node_hashes_dict = nx.weisfeiler_lehman_node_hashes(G, node_attr='feature', iterations=self.wl_iterations)
                    substructures = [h for hashes in node_hashes_dict.values() for h in hashes]
                except AttributeError:
                    # Fallback para versões antigas
                    substructures = []
                    for it in range(1, self.wl_iterations + 1):
                        substructures.append(nx.weisfeiler_lehman_graph_hash(G, node_attr='feature', iterations=it))
           
            # 2. Garantir que a lista nunca é vazia
            if not substructures:
                substructures = [f"node_count_{G.number_of_nodes()}"]
     
            tagged_data.append(TaggedDocument(words=substructures, tags=[str(i)]))
     
        # 3. Configuração do Doc2Vec para evitar o erro de 'high <= 0'
        # dm=0 seleciona PV-DBOW, que é mais rápido e não exige janela (window) de contexto
        self.model = Doc2Vec(
            vector_size=self.dimensions,
            dm=0,                # 0 = PV-DBOW (melhor para Graph2Vec)
            min_count=self.min_count,         # Não descarta nenhuma subestrutura rara
            workers=self.workers,
            seed=self.seed,
            epochs=self.epochs
        )
       
        print("A construir vocabulário...")
        self.model.build_vocab(tagged_data)
       
        print(f"A treinar em {len(tagged_data)} grafos...")
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
     
        self.embeddings = np.array([self.model.dv[str(i)] for i in range(len(graphs))])

    def get_embedding(self):
        return self.embeddings
 
if __name__ == '__main__':
    # Configuração de caminhos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    distance= 'cid'
    window_size = 15
    step_size = 1
    product_id = 907969
    enable_edges_within_star = False
    prefix = "" if enable_edges_within_star else "star_"
    pkl_path = os.path.join(
        current_dir,
        '..', 'GraphAnalysis', 'DynamicGraphPkls', distance, str(window_size), str(step_size), str(product_id),
        f'{prefix}dynamic_graphs_{distance}_Window{window_size}_Step{step_size}_pct0.5.pkl'
    )
 
    try:
        # 1. Carregar
        graphs_data = load_graphs(pkl_path, num_graphs=7)
       
        # 2. Gerar Embeddings
        # Aumentei as epochs para 50 para melhor convergência em amostras pequenas
        model = CustomGraph2Vec(dimensions=32, epochs=50)
        model.fit(graphs_data)
        vectors = model.get_embedding()
       
        # 3. Resultados
        print("\n--- Resultados ---")
        print(f"Matriz de embeddings: {vectors.shape}")
        print(f"Vetor do Grafo 0 (primeiros 5 elementos): {vectors[0][:5]}")
        print(f"Vetor do Grafo 1 (primeiros 5 elementos): {vectors[1][:5]}")
        print(f"Vetor do Grafo 2 (primeiros 5 elementos): {vectors[2][:5]}")
        print(f"Vetor do Grafo 3 (primeiros 5 elementos): {vectors[3][:5]}")
        print(f"Vetor do Grafo 4 (primeiros 5 elementos): {vectors[4][:5]}")
        print(f"Vetor do Grafo 5 (primeiros 5 elementos): {vectors[5][:5]}")
        # Teste de verificação:
        # Se correres este script duas vezes, os valores acima serão idênticos.
       
    except FileNotFoundError as e:
        print(f"Erro: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
 
 