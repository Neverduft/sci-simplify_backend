import urllib.request
import fitz
import re
import numpy as np
import tensorflow_hub as hub
import openai
import json
import os
from sklearn.neighbors import NearestNeighbors


def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace("\n", " ")
    text = re.sub("\s+", " ", text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(" ") for t in texts]
    page_nums = []
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i : i + word_length]
            if (
                (i + word_length) > len(words)
                and (len(chunk) < word_length)
                and (len(text_toks) != (idx + 1))
            ):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = " ".join(chunk).strip()
            chunk = f"[Page no. {idx+start_page}]" + " " + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    def __init__(self):
        self.use = hub.load("sentence-encoder") # https://tfhub.dev/google/universal-sentence-encoder/4
        self.fitted = False

    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i : (i + batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings


def load_recommender(path, start_page=1):
    global recommender
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return "Corpus Loaded."


def generate_text(openAI_key, prompt, model="gpt-3.5-turbo-0301"):
    openai.api_key = openAI_key
    completions = openai.ChatCompletion.create(
        model=model,
         messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7,
    )
    message = completions.choices[0].message['content']
    print(message)
    return message


def generate_answer(question, openAI_key):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += "search results:\n\n"
    for c in topn_chunks:
        prompt += c + "\n\n"

    prompt += (
        "Instructions: Compose a comprehensive reply to the query using the search results given. "
        "Cite each reference using [Page Number] notation (every result has this number at the beginning). "
        "Citation should be done at the end of each sentence. If the search results mention multiple subjects "
        "with the same name, create separate answers for each. Only include information found in the results and "
        "don't add any additional information. Make sure the answer is correct and don't output false content. "
        "If the text does not relate to the query, simply state 'Text Not Found in PDF'. Ignore outlier "
        "search results which has nothing to do with the question. Only answer what is asked. The "
        "answer should be short and concise. Answer step-by-step.\n\nQuery:\n{question}\nAnswer:\n"
    )

    answer = generate_text(openAI_key, prompt)
    return answer


def generate_blog_post_section(knowledge_level, section, openAI_key):
    question = "What is the section {section} about?"
    topn_chunks = recommender(question)
    prompt = (
        "Instructions: Reformat the following research paper section into a section of a blog post. Make the content understandable "
        "for someone with the knowledge level of {knowledge_level}. Use the [Page Number] notation for "
        "citations. Only include information found in the section and don't add any additional information. Make sure the content "
        "is correct and don't output false content. Ensure the content is clear, concise, organized, and accurate. Think and answer step by step.\n\n"
        "Knowledge Level: {knowledge_level}\n"
        "Research Paper section:\n\n"
    )

    if topn_chunks is not None:
        for c in topn_chunks:
            prompt += c + "\n\n"

    prompt += "\nOutput JSON:\n"

    answer = generate_text(openAI_key, prompt)
    return answer


def get_main_sections(openAI_key):
    question = "What are the main sections of this paper? The answer must consists out of only the titles of the sections separated by commas."

    topn_chunks = recommender(question)
    prompt = ""
    prompt += "search results:\n\n"
    for c in topn_chunks:
        prompt += c + "\n\n"

    prompt += (
        "Instructions: Compose a comprehensive reply to the query using the search results given. "
        "Only include information found in the results and "
        "don't add any additional information. Make sure the answer is correct and don't output false content. "
        "Ignore outlier search results which has nothing to do with the question. Only answer what is asked. The "
        "answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "
    )

    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text(openAI_key, prompt)
    return answer


 
    return main_sections


def question_answer(url, file_path, question, summarize, knowledge_level, openAI_key):
    if openAI_key.strip() == "":
        return "[ERROR]: Please enter you Open AI Key. Get your key here : https://platform.openai.com/account/api-keys"
    if url.strip() == "" and file_path == None:
        return "[ERROR]: Both URL and PDF is empty. Provide atleast one."

    if url.strip() != "" and file_path != None:
        return "[ERROR]: Both URL and PDF is provided. Please provide only one (eiter URL or PDF)."

    if url.strip() != "":
        glob_url = url
        download_pdf(glob_url, "corpus.pdf")
        load_recommender("corpus.pdf")

    else:
        load_recommender(file_path)

    if question.strip() == "" or summarize:
        main_sections = get_main_sections(openAI_key)
        main_sections_list = [
            section.strip() for section in main_sections.split(",") if section.strip()
        ]
        content = []
        for section in main_sections_list:
            content.append(
                generate_blog_post_section(knowledge_level, section, openAI_key)
            )
        
        # Save the content as a JSON object
        content_json = {"sections": main_sections_list, "content": content}
        
        # Save as file
        with open('test.json', 'w', encoding='utf-8') as f:
            json.dump(content_json, f, ensure_ascii=False, indent=4)

        return json.dumps(content_json, indent=2)
    else:   
        return generate_answer(question, openAI_key)


recommender = SemanticSearch()


# openai.api_key = os.getenv('Your_Key_Here')
