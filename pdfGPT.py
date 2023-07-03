import urllib.request
import fitz
import re
import numpy as np
import openai
import json
import os


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

    print(len(text_list))
    return text_list


def prepend_and_concat(string_list, prefix, end_page):
    # Start with an empty string
    result = ""

    # Ensure end_page is not greater than the length of string_list
    end_page = min(end_page, len(string_list))

    # Iterate over each string in the list up to end_page
    for count, string in enumerate(string_list[:end_page]):
        # Prepend the prefix and append a newline, then add to the result
        result += prefix + f" {count+1}: " + string + "\n"

    result = result.replace("’", "'")

    # Return the resulting string
    return result


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


# class SemanticSearch:
#     def __init__(self):
#         self.use = hub.load(
#             "sentence-encoder"
#         )  # Download model from here https://tfhub.dev/google/universal-sentence-encoder/4
#         self.fitted = False

#     def fit(self, data, batch=1000, n_neighbors=5):
#         self.data = data
#         self.embeddings = self.get_text_embedding(data, batch=batch)
#         n_neighbors = min(n_neighbors, len(self.embeddings))
#         self.nn = NearestNeighbors(n_neighbors=n_neighbors)
#         self.nn.fit(self.embeddings)
#         self.fitted = True

#     def __call__(self, text, return_data=True):
#         inp_emb = self.use([text])
#         neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

#         if return_data:
#             return [self.data[i] for i in neighbors]
#         else:
#             return neighbors

#     def get_text_embedding(self, texts, batch=1000):
#         embeddings = []
#         for i in range(0, len(texts), batch):
#             text_batch = texts[i : (i + batch)]
#             emb_batch = self.use(text_batch)
#             embeddings.append(emb_batch)
#         embeddings = np.vstack(embeddings)
#         return embeddings


def load_recommender(path, start_page=1):
    global recommender
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return "Corpus Loaded."


def generate_text(
    openAI_key, prompt, role="user", max_tokens=3000, model="gpt-3.5-turbo-16k"
):
    openai.api_key = openAI_key
    completions = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": role, "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )

    print(completions)
    message = completions.choices[0].message["content"]
    return message


def generate_answer(question, openAI_key):  # TODO Fix this prompt
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
        f"answer should be short and concise. Answer step-by-step.\n\nQuery:\n{question}\nAnswer:\n"
    )

    answer = generate_text(openAI_key, prompt)
    return answer


def generate_blog_post_section(knowledge_level, section, sections, openAI_key):
    question = f"What is the section '{section}' about?"  # TODO Add to prompt?

    topn_chunks = recommender(question)
    prompt = ""
    prompt += "Search results:\n\n"
    for c in topn_chunks:
        prompt += c + "\n\n"

        # TODO test sections [,...] current section: ,..

    prompt += (
        "Instructions: Using the given search results reformat the following section of a paper into a part of a blog post. "
        f"Make the content understandable for someone with the knowledge level of {knowledge_level}. "
        "Only include information found in the section and don't add any additional information. Make sure the content "
        "is correct and don't output false content. Ensure the content is clear, concise, organized, and accurate. Think step by step.\n\n"
        f"Research paper sections: {sections}\n"
        f"Research paper current section: {section}\n\n"
    )

    prompt += "Output:\n\n"

    answer = generate_text(openAI_key, prompt, max_tokens=500)
    return answer


def get_main_sections(openAI_key):
    question = "What are the main sections of this paper?"

    topn_chunks = recommender(question)
    prompt = ""
    prompt += "Search results:\n\n"
    for c in topn_chunks:
        prompt += c + "\n\n"

    prompt += (
        "Instructions: Compose a comprehensive reply to the query using the search results given. "
        "Only include information found in the results and "
        "don't add any additional information. Make sure the answer is correct and don't output false content. "
        "Ignore outlier search results which has nothing to do with the question. Only answer what is asked. "
        f"Think step by step. The answer must consist only out of titles of the sections separated by commas.\n\nQuery: {question}\n\nAnswer: "
    )

    print(prompt)

    main_sections = generate_text(openAI_key, prompt, "user")
    return main_sections


def generate_summary(knowledge_level, paper_text, openAI_key):
    prompt = (
        "Instructions: Given the following paper, please identify the main sections of the content and rewrite them along with any relevant subsections into parts of a summarized blog post. "
        f"Use a writing style typical for a blog post. Don't use repetetive sentence starts. "
        "Skip these sections in the paper: 'References', 'Acknowledgements' and 'Sources'. Put detailed focus on the main points of the paper and make sure each section is understandable and informative. You can combine multiple sections and subsections if appropriate. "
        "Only include information found in the paper and don't add any additional information. Make sure the content is correct and don't output false content. "
        "The output should be a JSON with two lists: 'section_titles' and 'sections'. Each index in the 'section_titles' list should correspond to the same index in the 'sections' list. Don't add the titles to the sections."
        "Use the following output format:\n"
        "{\n"
        '"section_titles": ["Section 1 title", "Section 2 title", "..."],\n'
        '"sections": ["Text", "Text 2", "..."],\n'
        "}\n\n"
        "Think step by step.\nText:\n\n"
    )

    prompt += paper_text

    #   " Using the given search results reformat the following section of a paper into a part of a blog post. "
    #     f"Make the content understandable for someone with the knowledge level of {knowledge_level}. "
    #     "Only include information found in the section and don't add any additional information. Make sure the content "
    #     "is correct and don't output false content. Ensure the content is clear, concise, organized, and accurate. Think step by step.\n\n"
    #     f"Research paper sections: {sections}\n"
    #     f"Research paper current section: {section}\n\n"

    prompt += "Output:\n\n"

    answer = generate_text(openAI_key, prompt, max_tokens=4000)
    return answer


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
        # load_recommender("corpus.pdf")
        string_list = pdf_to_text("corpus.pdf")

    else:
        # load_recommender(file_path)
        string_list = pdf_to_text(file_path)

    paper_text = prepend_and_concat(
        string_list=string_list, prefix="Page", end_page=len(string_list) - 1
    )

    if question.strip() == "debug":
        with open("test.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    elif question.strip() == "" or summarize:
        # main_sections = get_main_sections(openAI_key)
        # main_sections_list = [
        #     section.strip().replace(".", "")
        #     for section in main_sections.split(",")
        #     if section.strip()  # TODO fix prompt no references and no page numbers in blog
        # ]
        # content = []
        # print(main_sections_list)

        # for section in main_sections_list:
        #     content.append(
        #         generate_blog_post_section(
        #             knowledge_level, section, main_sections, openAI_key
        #         )
        #     )
        # break

        summary = generate_summary(knowledge_level, paper_text, openAI_key)

        print(summary)

        # Convert the string to a dictionary
        summary_dict = json.loads(summary)

        # # Save the content as a JSON object
        # content_json = {"sections": main_sections_list, "content": content}

        # Save as file
        with open("test.json", "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, ensure_ascii=False, indent=4)

        return json.dumps(summary_dict, indent=2)
    else:
        return generate_answer(question, openAI_key)


# recommender = SemanticSearch()

# openai.api_key = os.getenv('Your_Key_Here')
