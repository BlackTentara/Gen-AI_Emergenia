import streamlit as st
from duckduckgo_search import DDGS
from swarm import Swarm, Agent
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
MODEL = "llama3.2:latest"
client = Swarm()

st.set_page_config(page_title="AI News Processor", page_icon="📰")
st.title("📰 Breaking News")

def search_news(topic):
    """Search for news articles using DuckDuckGo"""
    with DDGS() as ddg:
        results = ddg.text(f"{topic} news {datetime.now().strftime('%Y-%m')}", max_results=3)
        if results:
            news_results = "\n\n".join([
                f"Title: {result['title']}\nURL: {result['href']}\nSummary: {result['body']}" 
                for result in results
            ])
            return news_results
        return f"No news found for {topic}."

# Create specialized agents
search_agent = Agent(
    name="Ahli pencari berita",
    instructions="""
    Anda adalah seorang spesialis pencarian berita kecelakaan. Tugas Anda adalah:
        1. Mencari berita paling relevan dan terbaru tentang topik kecelakaan yang diberikan.  
        2. Fokus pada berita yang membahas tentang Kecelakaan dan Bencana, seperti kecelakaan lalu lintar, gempa, banjir, dan sebagainya.  
        3. Memastikan hasil berasal dari sumber yang terpercaya.  
        4. Mengembalikan hasil pencarian mentah dalam format yang terstruktur.
    """,
    functions=[search_news],
    model=MODEL
)

synthesis_agent = Agent(
    name="Ahli sintesis berita",
    instructions="""
    Anda adalah seorang ahli sintesis berita tentang Kecelakaan dan Bencana. Tugas Anda adalah:
        1. Menganalisis artikel berita mentah terkait Kecelakaan dan Bencana yang diberikan.
        2. Menggabungkan informasi dari berbagai sumber.
        3. Membuat sintesis yang komprehensif namun ringkas.
        4. Berfokus pada fakta dan menjaga objektivitas jurnalistik.
        5. Hanya memberikan berita tentang Kecelakaan atau Bencana. Jika berita tersebut tidak tentang kecelakaan, nyatakan bahwa Anda tidak dapat memberikan informasi.
        6. Jika topik terkait seseorang, fokuslah pada menjelaskan Kecelakaan atau Bencana yang menimpa orang tersebut. Jika tidak ada informasi yang ditemukan, tulis "Tidak ada informasi yang ditemukan."

Berikan sintesis dalam 2-3 paragraf yang merangkum poin-poin utama.
    """,
    model=MODEL
)

summary_agent = Agent(
    name="Ahli perangkum berita",
    instructions="""
    Anda adalah seorang ahli dalam merangkum berita kecelakaan dan bencana.

    Tugas Anda:
        1. Informasi Utama:
           - Mulailah dengan perkembangan berita yang paling penting
           - Sertakan pihak-pihak terkait dan tindakan mereka
           - Tambahkan angka/data penting jika relevan
           - Jelaskan mengapa ini penting sekarang
           - Sebutkan implikasi langsungnya

        2. Pedoman Gaya:
           - Gunakan kata kerja yang kuat dan aktif
           - Bersikap spesifik, bukan umum
           - Jaga objektivitas jurnalistik
           - Manfaatkan setiap kata dengan maksimal
           - Jelaskan istilah teknis jika diperlukan

    Format: Buatlah paragraf tunggal antara 250-400 kata dalam BAHASA INDONESIA yang memberikan informasi dan menarik perhatian.
    Polanya: [Tanggal kejadian], [Judul Berita], [Berita Utama], [Detail Utama/Data]

    Fokuskan untuk menjawab: Apa yang terjadi? Mengapa ini signifikan? Apa dampaknya?

    PENTING: HANYA BERIKAN INFORMASI TENTANG KECELAKAAN DAN BENCANA, jika bukan tentang kecelakaan, katakan Anda tidak dapat memberikan informasi. Berikan HANYA paragraf ringkasan. Jangan sertakan frasa pengantar, label lebih dari satu, atau meta-teks seperti "Berikut adalah ringkasan" atau "Dalam gaya AP/Reuters." Mulailah langsung dengan isi berita.
    """,
    model=MODEL
)

def process_news(topic):
    """Run the news processing workflow"""
    with st.status("Processing news...", expanded=True) as status:
        # Search
        status.write("🔍 Searching for news...")
        search_response = client.run(
            agent=search_agent,
            messages=[{"role": "user", "content": f"Find news about {topic}"}]
        )
        raw_news = search_response.messages[-1]["content"]
        
        # Synthesize
        status.write("🔄 Synthesizing information...")
        synthesis_response = client.run(
            agent=synthesis_agent,
            messages=[{"role": "user", "content": f"Synthesize these news articles:\n{raw_news}"}]
        )
        synthesized_news = synthesis_response.messages[-1]["content"]
        
        # Summarize
        status.write("📝 Creating summary...")
        summary_response = client.run(
            agent=summary_agent,
            messages=[{"role": "user", "content": f"Summarize this synthesis:\n{synthesized_news}"}]
        )
        return raw_news, synthesized_news, summary_response.messages[-1]["content"]

# User Interface
topic = st.text_input("Beri topik berita:", value="")
if st.button("Cari Berita", type="primary"):
    if topic:
        try:
            raw_news, synthesized_news, final_summary = process_news(topic)
            st.header(f"📝 News Summary: {topic}")
            st.markdown(final_summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please enter a topic!")