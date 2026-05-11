import math

SKILL_ALIASES = {
    "python": "python", "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning", "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css", "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws",
    "android": "android", "firebase": "firebase",
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

RESUMES = [
    ("Arjun Sharma",    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair",      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta",     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel",     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh",    "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta",     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao",     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar",    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer",      "python, R, statistics, ML, regression, clustering, Power-BI"),
]

JOB_DESCRIPTIONS = [
    (
        "JD-1 — Kakao (ML Engineer)",
        "python machine_learning deep_learning tensorflow pytorch sql data_visualization "
        "nlp bert feature_engineering statistics"
    ),
    (
        "JD-2 — Naver (Backend Engineer)",
        "java spring_boot mysql postgresql microservices docker kubernetes "
        "rest_api ci_cd redis"
    ),
    (
        "JD-3 — Line (Frontend Engineer)",
        "javascript react vue typescript rest_api html_css "
        "nodejs graphql redux jest aws"
    ),
]

multi_word_keys = sorted(
    [k for k in SKILL_ALIASES if ' ' in k],
    key=lambda x: -len(x)
)

def normalize(raw_skills):
    tokens = [t.strip().lower() for t in raw_skills.split(',')]
    canonical = []
    for token in tokens:
        matched = False
        for phrase in multi_word_keys:
            if token == phrase:
                canonical.append(SKILL_ALIASES[phrase])
                matched = True
                break
        if not matched:
            if token in SKILL_ALIASES:
                canonical.append(SKILL_ALIASES[token])
    seen = set()
    deduped = []
    for s in canonical:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped

def build_vocabulary(normalized_resumes):
    vocab = set()
    for _, skills in normalized_resumes:
        vocab.update(skills)
    return sorted(vocab)

def compute_idf(vocab, normalized_resumes):
    N = len(normalized_resumes)
    df = {skill: 0 for skill in vocab}
    for _, skills in normalized_resumes:
        for s in skills:
            df[s] += 1
    return {skill: math.log(N / df[skill]) for skill in vocab}

def compute_tfidf_vector(skills, vocab, idf):
    n = len(skills)
    skill_set = set(skills)
    return [
        (1.0 / n) * idf[skill] if skill in skill_set else 0.0
        for skill in vocab
    ]

def build_jd_vector(jd_skills_str, vocab):
    jd_skills = set(jd_skills_str.split())
    return [1.0 if skill in jd_skills else 0.0 for skill in vocab]

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def main():
    normalized_resumes = [(name, normalize(raw)) for name, raw in RESUMES]

    print("=== STEP 1 & 2: Normalized + Deduplicated Skills ===")
    for name, skills in normalized_resumes:
        print(f"  {name}: {skills}")

    vocab = build_vocabulary(normalized_resumes)
    print(f"\n=== STEP 3: Vocabulary ({len(vocab)} skills) ===")
    print(f"  {vocab}")

    idf = compute_idf(vocab, normalized_resumes)
    print("\n=== STEP 4: IDF Values ===")
    for skill, val in idf.items():
        print(f"  {skill}: {val:.4f}")

    tfidf_vectors = [
        (name, compute_tfidf_vector(skills, vocab, idf))
        for name, skills in normalized_resumes
    ]

    print("\n=== STEP 5: JD Vectors + STEP 6: Rankings ===")
    print()
    for jd_name, jd_skills_str in JOB_DESCRIPTIONS:
        jd_vec = build_jd_vector(jd_skills_str, vocab)
        scores = [
            (name, cosine_similarity(vec, jd_vec))
            for name, vec in tfidf_vectors
        ]
        scores.sort(key=lambda x: (-x[1], x[0]))
        top3 = scores[:3]
        print(jd_name)
        print(", ".join(f"{name}({score:.2f})" for name, score in top3))
        print()

if __name__ == "__main__":
    main()
