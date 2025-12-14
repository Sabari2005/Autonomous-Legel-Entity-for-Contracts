from pyvis.network import Network

def create_combined_graph(chunks_data, output_file="combined_knowledge_graph.html"):
    net = Network(height="700px", width="100%", directed=True, notebook=True)
    net.force_atlas_2based()
    
    for chunk_idx, (chunk_text, risk_level, risk_category, similar_clauses) in enumerate(chunks_data):
        chunk_node_id = f"Chunk_{chunk_idx + 1}"
        risk_color = 'red' if risk_level.lower() == 'high' else 'orange' if risk_level.lower() == 'medium' else 'green'
        
        net.add_node(chunk_node_id, 
                    label=f"Chunk {chunk_idx + 1}",
                    color="lightblue",
                    title=f"{risk_level} risk - {risk_category}\n\n{chunk_text[:200]}...")
        
        risk_node = f"{chunk_node_id}_Risk_Level"
        category_node = f"{chunk_node_id}_Category"
        
        net.add_node(risk_node, 
                    label=f"Risk: {risk_level}",
                    color=risk_color,
                    title=f"Risk Level: {risk_level}")
        
        net.add_node(category_node,
                    label=f"Category: {risk_category}",
                    color="purple",
                    title=f"Category: {risk_category}")
        
        net.add_edge(chunk_node_id, risk_node, label="HAS_RISK")
        net.add_edge(chunk_node_id, category_node, label="HAS_CATEGORY")
        
        for i, clause in enumerate(similar_clauses[:3]):
            clause_id = f"{chunk_node_id}_Match_{i+1}"
            net.add_node(clause_id,
                        label=f"Match {i+1}",
                        color="lightcoral",
                        title=f"Similarity: {clause['similarity']:.2f}\n\n{clause['text'][:200]}...")
            net.add_edge(chunk_node_id, clause_id, label="SIMILAR_TO")
    
    net.show_buttons(filter_=["physics"])
    net.save_graph(output_file)
    return output_file