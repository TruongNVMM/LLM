# 🗂️ Kế Hoạch Dự Án: Xây Dựng & Triển Khai Mô Hình LLM Text-to-SQL

> **Mục tiêu:** Xây dựng ứng dụng web cho phép người dùng nhập câu hỏi bằng ngôn ngữ tự nhiên và nhận lại kết quả truy vấn SQL từ Databricks/Snowflake, tích hợp OpenAI GPT, triển khai trên AWS EC2 với HTTPS.

---

## 📋 Tổng Quan Dự Án

| Hạng mục | Chi tiết |
|---|---|
| **Ngôn ngữ** | Python |
| **Frontend** | Streamlit |
| **LLM** | OpenAI GPT-4 / GPT-3.5-turbo |
| **Database** | Databricks, Snowflake |
| **Cloud** | AWS EC2 |
| **Thời gian ước tính** | 6–8 tuần |

---

## 🗓️ PHASE 1 — Nền Tảng & Môi Trường (Tuần 1)

### Bước 1.1 — Nắm vững kiến thức LLM cơ bản

**Mục tiêu:** Hiểu LLM là gì, cách hoạt động và ứng dụng cho NLP.

**Việc cần làm:**
- Đọc tài liệu: "What are Large Language Models?" (OpenAI, Anthropic docs)
- Tìm hiểu khái niệm: Tokenization, Prompt, Completion, Temperature, Max Tokens
- Thực hành gọi OpenAI API bằng Python (trả lời câu hỏi đơn giản)
- Hiểu sự khác biệt giữa `chat/completions` và `completions` endpoint

**Kết quả đạt được:**
- Giải thích được LLM hoạt động như thế nào
- Gọi được OpenAI API và nhận phản hồi từ model

---

### Bước 1.2 — Làm quen công nghệ dự án

**Mục tiêu:** Hiểu vai trò từng công nghệ trong hệ thống.

**Việc cần làm:**
- **Streamlit:** Đọc docs chính thức, tạo app "Hello World", tìm hiểu `st.text_input`, `st.dataframe`, `st.session_state`
- **OpenAI Python SDK:** Cài đặt `openai`, thực hành `client.chat.completions.create()`
- **Databricks:** Tạo tài khoản Community Edition, tìm hiểu Workspace, Cluster, Notebook, SQL Warehouse
- **Snowflake:** Tạo tài khoản trial, tìm hiểu Database, Schema, Warehouse, Role

**Tài nguyên:**
```
https://docs.streamlit.io/get-started
https://platform.openai.com/docs/quickstart
https://docs.databricks.com/getting-started
https://docs.snowflake.com/en/user-guide-getting-started
```

---

### Bước 1.3 — Thiết lập môi trường phát triển

**Mục tiêu:** Cấu hình môi trường dev chuẩn, tái sử dụng được.

**Việc cần làm:**

1. **Cài đặt VS Code + extensions:**
   - Python extension
   - Pylance
   - GitLens
   - `.env` support

2. **Tạo virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows
pip install streamlit openai databricks-sql-connector snowflake-connector-python python-dotenv
pip freeze > requirements.txt
```

3. **Cấu hình `.env` file:**
```
OPENAI_API_KEY=sk-...
DATABRICKS_HOST=https://...
DATABRICKS_TOKEN=dapi...
DATABRICKS_HTTP_PATH=/sql/...
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_ACCOUNT=...
```

4. **Tạo `.gitignore`** để bảo vệ secrets:
```
.env
.venv/
__pycache__/
*.pyc
```

5. **Khởi tạo Git repo** và commit lần đầu.

**Kết quả đạt được:**
- Chạy được `streamlit run app.py` không lỗi
- VS Code nhận diện đúng Python interpreter trong `.venv`

---

## 🗓️ PHASE 2 — Kết Nối Database & Query Cơ Bản (Tuần 2)

### Bước 2.1 — Kết nối an toàn với Databricks

**Mục tiêu:** Kết nối và truy vấn dữ liệu từ Databricks SQL Warehouse.

**Việc cần làm:**

1. Tạo Personal Access Token trên Databricks UI
2. Lấy HTTP Path của SQL Warehouse
3. Viết module `db/databricks_connector.py`:

```python
from databricks import sql
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )

def execute_query(query: str) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
```

4. Test kết nối với query đơn giản: `SELECT current_timestamp()`
5. Tải lên 2–3 dataset mẫu (CSV) vào Databricks để dùng cho dự án

**Lưu ý bảo mật:**
- Không bao giờ hardcode token trong code
- Sử dụng `python-dotenv` để load từ `.env`
- Đặt token có expiry ngắn trong môi trường dev

---

### Bước 2.2 — Lấy Schema tự động từ Database

**Mục tiêu:** Cho LLM biết cấu trúc database để sinh SQL chính xác.

**Việc cần làm:**

Viết hàm `get_schema_info()` trả về mô tả tất cả bảng và cột:

```python
def get_schema_info(database: str, schema: str) -> str:
    query = f"""
    SELECT table_name, column_name, data_type, comment
    FROM {database}.information_schema.columns
    WHERE table_schema = '{schema}'
    ORDER BY table_name, ordinal_position
    """
    rows = execute_query(query)
    
    schema_text = ""
    current_table = None
    for row in rows:
        if row["table_name"] != current_table:
            current_table = row["table_name"]
            schema_text += f"\nTable: {current_table}\n"
        schema_text += f"  - {row['column_name']} ({row['data_type']})"
        if row.get("comment"):
            schema_text += f" -- {row['comment']}"
        schema_text += "\n"
    return schema_text
```

**Kết quả đạt được:**
- In ra được schema của toàn bộ database dưới dạng văn bản
- Hiểu cách information_schema hoạt động

---

## 🗓️ PHASE 3 — LLM Core: Prompt Engineering & Text-to-SQL (Tuần 3)

### Bước 3.1 — Tích hợp OpenAI API

**Mục tiêu:** Gọi OpenAI API và nhận SQL từ câu hỏi ngôn ngữ tự nhiên.

**Việc cần làm:**

Viết module `llm/openai_client.py`:

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_sql(
    user_question: str,
    schema_info: str,
    model: str = "gpt-4"
) -> str:
    system_prompt = f"""You are an expert SQL analyst. 
Given the following database schema:

{schema_info}

Generate a valid SQL query to answer the user's question.
Rules:
- Return ONLY the SQL query, no explanation
- Use proper table and column names from the schema
- Use LIMIT 100 by default unless specified
- Handle NULL values appropriately
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()
```

---

### Bước 3.2 — Prompt Engineering nâng cao

**Mục tiêu:** Thiết kế prompt tạo ra SQL chính xác, phức tạp và ổn định.

**Kỹ thuật cần nắm:**

1. **Few-shot prompting** — cung cấp ví dụ câu hỏi → SQL:
```python
FEW_SHOT_EXAMPLES = """
Example 1:
Question: How many orders were placed last month?
SQL: SELECT COUNT(*) as total_orders FROM orders WHERE order_date >= date_trunc('month', current_date - interval '1 month') AND order_date < date_trunc('month', current_date)

Example 2:
Question: Top 5 customers by revenue
SQL: SELECT customer_id, SUM(amount) as total_revenue FROM orders GROUP BY customer_id ORDER BY total_revenue DESC LIMIT 5
"""
```

2. **Chain-of-thought** — yêu cầu model suy nghĩ trước khi viết SQL:
```
"First identify the relevant tables, then determine the joins needed, then write the SQL"
```

3. **Output format constraints:**
```
"Return the SQL in a code block wrapped with ```sql ... ```"
```

4. **Negative examples** — nói rõ những gì KHÔNG được làm:
```
"Do NOT use table names that are not in the schema above"
"Do NOT generate DROP, DELETE, UPDATE, or INSERT statements"
```

**Việc cần làm:**
- Test ít nhất 20 câu hỏi tự nhiên khác nhau (đơn giản → phức tạp)
- Ghi lại các câu fail và cải thiện prompt
- So sánh kết quả với `temperature=0` vs `temperature=0.2`

---

### Bước 3.3 — Xử lý lỗi SQL mạnh mẽ

**Mục tiêu:** Ứng dụng không crash khi LLM sinh ra SQL sai.

**Việc cần làm:**

```python
import re

def clean_sql(raw_output: str) -> str:
    """Trích xuất SQL thuần từ output của LLM"""
    # Xóa markdown code blocks
    sql = re.sub(r'```sql\s*', '', raw_output)
    sql = re.sub(r'```', '', sql).strip()
    return sql

def validate_sql(sql: str) -> tuple[bool, str]:
    """Kiểm tra SQL không chứa các lệnh nguy hiểm"""
    dangerous = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE']
    upper_sql = sql.upper()
    for keyword in dangerous:
        if keyword in upper_sql:
            return False, f"SQL contains forbidden keyword: {keyword}"
    return True, "OK"

def safe_execute(sql: str) -> tuple[list, str | None]:
    """Thực thi SQL với error handling đầy đủ"""
    try:
        cleaned = clean_sql(sql)
        is_valid, msg = validate_sql(cleaned)
        if not is_valid:
            return [], msg
        results = execute_query(cleaned)
        return results, None
    except Exception as e:
        return [], f"Query execution error: {str(e)}"
```

---

## 🗓️ PHASE 4 — Self-Correction Loop & Tối Ưu (Tuần 4)

### Bước 4.1 — Vòng phản hồi tự điều chỉnh (Self-Correcting Loop)

**Mục tiêu:** Khi SQL lỗi, tự động yêu cầu LLM sửa lại.

**Việc cần làm:**

```python
def generate_sql_with_retry(
    user_question: str,
    schema_info: str,
    max_retries: int = 3
) -> tuple[str, list, str | None]:
    
    sql = generate_sql(user_question, schema_info)
    conversation_history = [
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": sql}
    ]
    
    for attempt in range(max_retries):
        results, error = safe_execute(sql)
        
        if error is None:
            return sql, results, None  # Thành công
        
        # Gửi lỗi cho LLM để sửa
        correction_prompt = f"""The SQL query you generated produced an error:

Error: {error}

Please fix the SQL query. Return ONLY the corrected SQL."""
        
        conversation_history.append({"role": "user", "content": correction_prompt})
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=conversation_history,
            temperature=0
        )
        
        sql = clean_sql(response.choices[0].message.content)
        conversation_history.append({"role": "assistant", "content": sql})
    
    return sql, [], f"Failed after {max_retries} attempts. Last error: {error}"
```

---

### Bước 4.2 — Xác thực và tối ưu truy vấn SQL

**Mục tiêu:** Kiểm tra và cải thiện hiệu suất các query phức tạp.

**Kỹ thuật:**
- Thêm `EXPLAIN` trước query để xem query plan
- Kiểm tra missing indexes, full table scans
- Thêm timeout cho query (tránh query vô tận)
- Log slow queries (> 5 giây) để review

```python
def analyze_query_performance(sql: str) -> dict:
    explain_sql = f"EXPLAIN {sql}"
    try:
        plan = execute_query(explain_sql)
        return {"plan": plan, "warning": None}
    except Exception as e:
        return {"plan": None, "warning": str(e)}
```

---

## 🗓️ PHASE 5 — Streamlit UI & Tính Năng Nâng Cao (Tuần 5)

### Bước 5.1 — Xây dựng giao diện Streamlit

**Mục tiêu:** Tạo UI trực quan, dễ sử dụng.

**Cấu trúc file:**
```
project/
├── app.py                    # Entry point Streamlit
├── pages/
│   ├── 1_Query.py           # Trang truy vấn chính
│   ├── 2_ERD.py             # Trang xem ERD
│   └── 3_History.py         # Lịch sử truy vấn
├── llm/
│   ├── openai_client.py
│   └── prompt_templates.py
├── db/
│   ├── databricks_connector.py
│   └── schema_extractor.py
├── auth/
│   └── authenticator.py
├── utils/
│   └── sql_utils.py
├── .env
└── requirements.txt
```

**Nội dung `app.py`:**
```python
import streamlit as st

st.set_page_config(
    page_title="Text-to-SQL Assistant",
    page_icon="🔍",
    layout="wide"
)

# Authentication check
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.switch_page("pages/Login.py")
```

**Tính năng trang Query chính:**
- Text area nhập câu hỏi tiếng Việt / Anh
- Hiển thị SQL được sinh ra với syntax highlighting
- Hiển thị kết quả dạng dataframe có thể lọc, sort
- Nút "Download CSV"
- Hiển thị thời gian thực thi query
- Lưu lịch sử query vào `st.session_state`

---

### Bước 5.2 — Xác thực người dùng & quản lý phiên

**Mục tiêu:** Bảo vệ ứng dụng, quản lý session an toàn.

**Việc cần làm:**

```python
# auth/authenticator.py
import hashlib
import os

USERS = {
    "admin": hashlib.sha256("password123".encode()).hexdigest(),
}

def authenticate(username: str, password: str) -> bool:
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return USERS.get(username) == hashed

# Trong Streamlit page:
def login_page():
    st.title("🔐 Đăng nhập")
    username = st.text_input("Tên đăng nhập")
    password = st.text_input("Mật khẩu", type="password")
    
    if st.button("Đăng nhập"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Sai tên đăng nhập hoặc mật khẩu")
```

**Quản lý session:**
- Lưu username, query history trong `st.session_state`
- Thêm nút "Đăng xuất" xóa session
- Set session timeout (option nâng cao)

---

### Bước 5.3 — Tạo sơ đồ ERD động bằng LLM

**Mục tiêu:** Tự động sinh và hiển thị ERD từ schema database.

**Việc cần làm:**

1. Dùng LLM để sinh Mermaid diagram code từ schema:

```python
def generate_erd(schema_info: str) -> str:
    prompt = f"""Given this database schema:

{schema_info}

Generate a Mermaid ERD diagram showing tables, columns, and relationships.
Use erDiagram syntax. Return ONLY the Mermaid code."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
```

2. Hiển thị trong Streamlit:

```python
import streamlit.components.v1 as components

mermaid_code = generate_erd(schema_info)

components.html(f"""
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<div class="mermaid">
{mermaid_code}
</div>
<script>mermaid.initialize({{startOnLoad:true}});</script>
""", height=600)
```

---

## 🗓️ PHASE 6 — Testing & Kiểm Thử Toàn Diện (Tuần 6)

### Bước 6.1 — Unit Testing

**Mục tiêu:** Đảm bảo từng module hoạt động đúng độc lập.

```python
# tests/test_sql_utils.py
import pytest
from utils.sql_utils import clean_sql, validate_sql

def test_clean_sql_removes_markdown():
    raw = "```sql\nSELECT * FROM users\n```"
    assert clean_sql(raw) == "SELECT * FROM users"

def test_validate_sql_blocks_dangerous():
    is_valid, msg = validate_sql("DROP TABLE users")
    assert is_valid == False
    assert "DROP" in msg

def test_validate_sql_allows_select():
    is_valid, msg = validate_sql("SELECT * FROM orders LIMIT 10")
    assert is_valid == True
```

**Chạy test:**
```bash
pip install pytest
pytest tests/ -v
```

---

### Bước 6.2 — Kiểm thử SQL phức tạp

**Việc cần thực hành với ít nhất 30 test case:**

| Loại query | Ví dụ câu hỏi |
|---|---|
| Aggregation | "Tổng doanh thu theo tháng năm 2024" |
| JOIN | "Danh sách khách hàng và số đơn hàng của họ" |
| Subquery | "Khách hàng có doanh thu cao hơn mức trung bình" |
| Window function | "Xếp hạng doanh thu theo khu vực" |
| Date filter | "Đơn hàng trong 30 ngày gần nhất" |
| NULL handling | "Sản phẩm chưa có giá" |
| Complex JOIN | "Doanh thu theo danh mục sản phẩm và khu vực" |

**Ghi lại kết quả** vào file `test_results.md`:
- Câu hỏi
- SQL được sinh ra
- Kết quả có đúng không
- Điều chỉnh prompt nếu cần

---

## 🗓️ PHASE 7 — Triển Khai AWS EC2 với HTTPS (Tuần 7–8)

### Bước 7.1 — Chuẩn bị AWS

**Mục tiêu:** Tạo EC2 instance và cấu hình cơ bản.

**Việc cần làm:**

1. Tạo tài khoản AWS (hoặc dùng Free Tier)
2. Launch EC2 instance:
   - AMI: Ubuntu Server 22.04 LTS
   - Instance type: `t2.medium` (2 CPU, 4GB RAM)
   - Storage: 20GB SSD
   - Security Group: mở port 22 (SSH), 80 (HTTP), 443 (HTTPS)
3. Tạo và download Key Pair (`.pem` file)
4. Gắn Elastic IP để có IP tĩnh
5. Tạo domain (hoặc dùng IP cho môi trường dev)

---

### Bước 7.2 — Deploy ứng dụng lên EC2

**Việc cần làm:**

```bash
# 1. SSH vào server
ssh -i "key.pem" ubuntu@<EC2_PUBLIC_IP>

# 2. Update hệ thống
sudo apt update && sudo apt upgrade -y

# 3. Cài Python
sudo apt install python3-pip python3-venv git -y

# 4. Clone code từ GitHub
git clone https://github.com/your-repo/text-to-sql.git
cd text-to-sql

# 5. Tạo venv và cài dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 6. Tạo .env file trên server (KHÔNG commit lên git)
nano .env
# Điền các biến môi trường

# 7. Test chạy thử
streamlit run app.py --server.port 8501
```

---

### Bước 7.3 — Cấu hình Nginx + SSL (HTTPS)

**Mục tiêu:** Ứng dụng chạy HTTPS qua domain.

```bash
# 1. Cài Nginx
sudo apt install nginx -y

# 2. Cấu hình reverse proxy
sudo nano /etc/nginx/sites-available/text-to-sql

# Nội dung config:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}

sudo ln -s /etc/nginx/sites-available/text-to-sql /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# 3. Cài Let's Encrypt SSL (miễn phí)
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
# Certbot tự động cấu hình HTTPS và auto-renew
```

---

### Bước 7.4 — Chạy ứng dụng như service (Systemd)

**Mục tiêu:** Ứng dụng tự khởi động lại khi server reboot.

```bash
sudo nano /etc/systemd/system/text-to-sql.service

# Nội dung:
[Unit]
Description=Text-to-SQL Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/text-to-sql
Environment="PATH=/home/ubuntu/text-to-sql/.venv/bin"
ExecStart=/home/ubuntu/text-to-sql/.venv/bin/streamlit run app.py --server.port 8501 --server.headless true
Restart=always

[Install]
WantedBy=multi-user.target

# Kích hoạt service
sudo systemctl daemon-reload
sudo systemctl enable text-to-sql
sudo systemctl start text-to-sql
sudo systemctl status text-to-sql
```

---

### Bước 7.5 — Bảo mật & Khả năng mở rộng

**Best practices cần áp dụng:**

**Bảo mật:**
- [ ] Không expose port 8501 ra internet (chỉ qua Nginx)
- [ ] Cấu hình AWS Security Group chỉ mở port 80, 443, 22
- [ ] Thêm rate limiting trên Nginx (tránh spam API)
- [ ] Rotate API keys định kỳ
- [ ] Dùng AWS Secrets Manager thay `.env` cho production
- [ ] Enable AWS CloudTrail để audit logs
- [ ] Cài `fail2ban` để chặn brute force SSH

**Khả năng mở rộng:**
- [ ] Dùng Application Load Balancer khi cần scale ngang
- [ ] Cache schema info (không fetch mỗi request)
- [ ] Dùng Redis để cache kết quả query giống nhau
- [ ] Cân nhắc dùng AWS RDS thay file-based session storage

```python
# Cache schema với TTL 10 phút
import functools
import time

_schema_cache = {}

def get_cached_schema(database: str, schema: str) -> str:
    cache_key = f"{database}.{schema}"
    now = time.time()
    
    if cache_key in _schema_cache:
        cached_value, cached_time = _schema_cache[cache_key]
        if now - cached_time < 600:  # 10 phút
            return cached_value
    
    schema_info = get_schema_info(database, schema)
    _schema_cache[cache_key] = (schema_info, now)
    return schema_info
```

---

## 📁 Cấu Trúc Thư Mục Cuối Cùng

```
text-to-sql/
├── app.py
├── pages/
│   ├── Login.py
│   ├── 1_Query.py
│   ├── 2_ERD.py
│   └── 3_History.py
├── llm/
│   ├── __init__.py
│   ├── openai_client.py
│   └── prompt_templates.py
├── db/
│   ├── __init__.py
│   ├── databricks_connector.py
│   ├── snowflake_connector.py
│   └── schema_extractor.py
├── auth/
│   ├── __init__.py
│   └── authenticator.py
├── utils/
│   ├── __init__.py
│   └── sql_utils.py
├── tests/
│   ├── test_sql_utils.py
│   └── test_openai_client.py
├── .env                   # KHÔNG commit lên git
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ✅ Checklist Kết Quả Đạt Được

Sau khi hoàn thành, đảm bảo tất cả mục sau được tích:

- [ ] Giải thích được LLM và ứng dụng của nó cho NLP
- [ ] Sử dụng thành thạo Streamlit, OpenAI, Databricks, Snowflake
- [ ] Tạo và quản lý được virtual environment Python
- [ ] Kết nối an toàn với Databricks bằng token
- [ ] Tích hợp OpenAI API để chuyển đổi NL → SQL
- [ ] Nắm vững prompt engineering và prompt chaining
- [ ] Xây dựng error handling mạnh mẽ cho SQL execution
- [ ] Tối ưu và xác thực truy vấn SQL
- [ ] Xác thực người dùng và quản lý session trong Streamlit
- [ ] Kiểm thử SQL phức tạp đảm bảo độ chính xác
- [ ] Tạo ERD động bằng LLM + Mermaid
- [ ] Triển khai self-correcting loop cho SQL generation
- [ ] Deploy thành công lên AWS EC2 với HTTPS
- [ ] Áp dụng best practices bảo mật và scalability

---

## 📚 Tài Nguyên Tham Khảo

| Tài nguyên | Link |
|---|---|
| OpenAI API Docs | https://platform.openai.com/docs |
| Streamlit Docs | https://docs.streamlit.io |
| Databricks SQL Connector | https://docs.databricks.com/dev-tools/python-sql-connector.html |
| Snowflake Connector | https://docs.snowflake.com/en/user-guide/python-connector |
| AWS EC2 Getting Started | https://docs.aws.amazon.com/ec2/index.html |
| Let's Encrypt Certbot | https://certbot.eff.org |
| Nginx Reverse Proxy | https://nginx.org/en/docs |
| Mermaid ERD Syntax | https://mermaid.js.org/syntax/entityRelationshipDiagram.html |

---

*Kế hoạch được xây dựng để đảm bảo từng bước tích lũy kỹ năng và có sản phẩm chạy được sau mỗi phase. Điều chỉnh timeline theo tiến độ thực tế.*
