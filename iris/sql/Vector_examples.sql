
/* Simple example using Vector data type */
/*   
 * Typically if you are using the Vector type directly, you will be doing this from the Python API
 * and using a library or API to create the embeddings
 */

--Drop table VectorExample

Create table VectorExample (name varchar(100), name_vector VECTOR(float, 7))

Insert into VectorExample (name, name_vector) values ('Patrick', TO_VECTOR('[0.25, .10, .05, .04, .01, .05, .50]') )

Insert into VectorExample (name, name_vector) values ('Talvia', TO_VECTOR('[0.333, .10, .15, .04, .03, .1769, .1701]') )

Select * from VectorExample

Select name, VECTOR_COSINE(TO_VECTOR('[0.233, .12, .03, .07, .08, .05, .4170]'), name_vector) as Cosine,
        VECTOR_DOT_PRODUCT(TO_VECTOR('[0.233, .12, .03, .07, .08, .05, .4170]'), name_vector) as DotProd
        from VectorExample 

        
/* Simple example using Embedding data type */
/*
 * The Embedding data type and function allow you to use your embedding library or API without leaving SQL
 * For this example
 */

-- Drop table VectorExample
-- The embedding model must first be added to the config table. This also requires that all required libs are installed
INSERT INTO %Embedding.Config (Name, Configuration, EmbeddingClass, Description)
 VALUES ('sentence-transformers-config',
          '{"modelName":"sentence-transformers/all-MiniLM-L6-v2",
             "hfCachePath":"/Users/InterSystems/VEC147/hfCache",
             "maxTokens": 256,
             "checkTokenCount": true}',
          '%Embedding.SentenceTransformers',
          '',
          'a small SentenceTransformers embedding model')
          
Create table EmbeddingExample (name varchar(100), name_embedding EMBEDDING('sentence-transformers/all-MiniLM-L6-v2','name'))

Insert into EmbeddingExample (name) values ('Patrick') 

Insert into EmbeddingExample (name) values ('Thomas')

Select name, VECTOR_COSINE(name_embedding, Embedding('Pat')), VECTOR_DOT_PRODUCT(name_embedding, Embedding('Pat'))
from EmbeddingExample


/* You can also create an ANN(Approximate Nearest Neighbors) index on Embedding and Vector columns
 * This index will be used by the SQL optimizer in qeuries that use a TOP clause and an ORDER BY ... DESC 
 */ 

CREATE INDEX HNSWIndex ON TABLE VectorExample (name_vector) AS HNSW(Distance='Cosine')
  
CREATE INDEX HNSWIndex ON TABLE EmbeddingExample (name_embedding) AS HNSW(Distance='Cosine')
  