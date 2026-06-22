import re 


class TextChunker:
    
    def __init__(self, chunk_size , overlap=50):
        self.chunk_size = chunk_size 
        self.overlap = overlap
        
    def chunk_by_word(self,text):
        
        words = text.split()
        chunks = []
        for i in range ( 0 , len(words) , self.chunk_size - self.overlap):
            chunk = ''.join(words[ i : i + self.chunk_size])
            chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break
        return chunks
    
    
    def chunk_by_paragraphs(self, text):
        """A simple semantic chunker relying on double newlines."""
        paragraphs = re.split(r'\n\n+', text.strip())
        return [p.strip() for p in paragraphs if len(p.strip()) > 10] 
    

