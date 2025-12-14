from neo4j import GraphDatabase
from config import uri, username, password

driver = GraphDatabase.driver(uri, auth=(username, password))

def fetch_all_clauses():
    with driver.session() as session:
        result = session.run("MATCH (c:Clause) RETURN c.text AS text, c.risk_level AS risk_level, c.risk_category AS risk_category")
        return [record.data() for record in result]