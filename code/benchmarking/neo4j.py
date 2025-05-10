"""
Neo4j Semantic Querying System
This script loads entity-relation-entity triples from a CSV file into Neo4j
and provides a natural language interface for querying the knowledge graph.
"""

from neo4j import GraphDatabase
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import numpy as np

class Neo4jSemanticQuery:
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="password"):
        """Initialize connection to Neo4j database"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        # Load sentence transformer model for semantic matching
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Store relation embeddings for semantic matching
        self.relation_embeddings = {}
        self.relations = []

    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()

    def load_csv_triples(self, csv_path, subject_col="subject", relation_col="relation", object_col="object"):
        """
        Load triples from a CSV file and create graph in Neo4j
        
        Args:
            csv_path (str): Path to CSV file
            subject_col (str): Column name for subject entities
            relation_col (str): Column name for relations
            object_col (str): Column name for object entities
        """
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Clear existing data (optional)
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        # Create unique constraints (indices)
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")
        
        # Process and create entities and relationships
        with self.driver.session() as session:
            # Create a unique set of all entities
            all_entities = set(df[subject_col]).union(set(df[object_col]))
            
            # Batch create all entities
            for entity in all_entities:
                session.run(
                    "MERGE (e:Entity {name: $name})",
                    name=entity
                )
            
            # Create relationships
            for _, row in df.iterrows():
                session.run(
                    """
                    MATCH (s:Entity {name: $subject})
                    MATCH (o:Entity {name: $object})
                    MERGE (s)-[r:`" + row[relation_col] + "`]->(o)
                    """,
                    subject=row[subject_col],
                    object=row[object_col]
                )
            
            # Store all unique relations for semantic matching
            unique_relations = df[relation_col].unique()
            self.relations = list(unique_relations)
            
            # Create embeddings for relations
            for relation in self.relations:
                self.relation_embeddings[relation] = self.model.encode(relation)
            
            print(f"Loaded {len(all_entities)} entities and {len(self.relations)} relation types.")

    def find_closest_relation(self, query_text, top_n=3):
        """Find the closest relation types to the query using semantic similarity"""
        # Generate embedding for the query
        query_embedding = self.model.encode(query_text)
        
        similarities = {}
        for relation, embedding in self.relation_embeddings.items():
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities[relation] = similarity
        
        # Sort by similarity (descending)
        sorted_relations = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_relations[:top_n]

    def extract_entities_from_query(self, query):
        """
        Extract potential entity names from the query
        This is a simple implementation and could be enhanced
        """
        # Look for capitalized words or words in quotes as potential entities
        entity_pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)|"([^"]+)"|\'([^\']+)\''
        matches = re.findall(entity_pattern, query)
        
        # Flatten and clean the matches
        entities = []
        for match in matches:
            for group in match:
                if group and len(group) > 1:  # Avoid single character matches
                    entities.append(group)
        
        return list(set(entities))

    def semantic_query(self, query_text):
        """
        Process a natural language query and return relevant triples
        
        Args:
            query_text (str): Natural language query
            
        Returns:
            list: Relevant triples from the knowledge graph
        """
        # Find potential entities in the query
        potential_entities = self.extract_entities_from_query(query_text)
        
        # Find semantically similar relations
        similar_relations = self.find_closest_relation(query_text)
        
        results = []
        with self.driver.session() as session:
            # If we found specific entities, use them in the query
            if potential_entities:
                for entity in potential_entities:
                    # Try entity as subject
                    for relation, similarity in similar_relations:
                        result = session.run(
                            f"""
                            MATCH (s:Entity)-[r:`{relation}`]->(o:Entity)
                            WHERE s.name CONTAINS $entity OR o.name CONTAINS $entity
                            RETURN s.name as subject, type(r) as relation, o.name as object, 
                                   {similarity} as similarity
                            LIMIT 10
                            """,
                            entity=entity
                        )
                        results.extend(result.data())
            
            # If no results yet, try more general query with just the relation types
            if not results:
                for relation, similarity in similar_relations:
                    result = session.run(
                        f"""
                        MATCH (s:Entity)-[r:`{relation}`]->(o:Entity)
                        RETURN s.name as subject, type(r) as relation, o.name as object,
                               {similarity} as similarity
                        LIMIT 10
                        """
                    )
                    results.extend(result.data())
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Limit and format results
        results = results[:10]  # Limit to top 10
        formatted_results = []
        for r in results:
            formatted_results.append((r['subject'], r['relation'], r['object']))
            
        return formatted_results

# Example usage
if __name__ == "__main__":
    # Example CSV content (you can replace with your file path)
    example_csv = """subject,relation,object
    Alice,KNOWS,Bob
    Bob,WORKS_AT,Acme Corp
    Charlie,LIVES_IN,New York
    Alice,VISITED,Paris
    Bob,FRIEND_OF,Charlie
    Paris,LOCATED_IN,France
    Acme Corp,HEADQUARTERED_IN,San Francisco
    """
    
    # Save example to a temporary CSV
    with open("example_triples.csv", "w") as f:
        f.write(example_csv)
    
    # Create the semantic query engine
    query_engine = Neo4jSemanticQuery()
    
    try:
        # Load data
        query_engine.load_csv_triples("example_triples.csv")
        
        # Example queries
        print("\nQuery 1: Where does Bob work?")
        results = query_engine.semantic_query("Where does Bob work?")
        for subj, rel, obj in results:
            print(f"{subj} -- {rel} --> {obj}")
            
        print("\nQuery 2: Who knows whom?")
        results = query_engine.semantic_query("Who knows whom?")
        for subj, rel, obj in results:
            print(f"{subj} -- {rel} --> {obj}")
            
        print("\nQuery 3: Tell me about Paris")
        results = query_engine.semantic_query("Tell me about Paris")
        for subj, rel, obj in results:
            print(f"{subj} -- {rel} --> {obj}")
        
        # Interactive query mode
        print("\nEnter your questions (type 'exit' to quit):")
        while True:
            user_query = input("> ")
            if user_query.lower() == 'exit':
                break
                
            results = query_engine.semantic_query(user_query)
            if results:
                print("Results:")
                for subj, rel, obj in results:
                    print(f"{subj} -- {rel} --> {obj}")
            else:
                print("No relevant information found.")
                
    finally:
        query_engine.close()