"""
Synthetic data generator for the AI Recruiter Ranking System.

This stands in for the real hackathon dataset. It produces:
  - candidates.csv   : ~120 candidate profiles with career history, skills,
                        behavioral signals, and platform activity
  - job_description.txt : one realistic JD to rank candidates against

Swap this out with the real dataset by matching the column schema documented
in README.md -> "Data Schema". The rest of the pipeline does not need to change.
"""

import csv
import json
import random
from pathlib import Path

random.seed(42)

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Building blocks for generating varied, realistic profiles
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sara", "Karan", "Meera",
    "Arjun", "Divya", "Nikhil", "Isha", "Rahul", "Tanya", "Aditya", "Pooja",
    "Sanjay", "Neha", "Varun", "Riya", "Amit", "Shreya", "Kabir", "Anika",
    "Dev", "Ritika", "Suresh", "Kavya", "Manish", "Lakshmi", "Jay", "Nisha",
    "Harsh", "Simran", "Vivek", "Pallavi", "Ashish", "Swati", "Gaurav", "Rina",
    "Faisal", "Megha", "Rajat", "Sneha", "Imran", "Anjali", "Tarun", "Komal",
]
LAST_NAMES = [
    "Sharma", "Verma", "Iyer", "Patel", "Nair", "Gupta", "Reddy", "Joshi",
    "Mehta", "Singh", "Rao", "Kapoor", "Chatterjee", "Bose", "Pillai", "Das",
    "Khan", "Bansal", "Agarwal", "Menon", "Pandey", "Saxena", "Malhotra", "Sinha",
]

COMPANIES_BIGTECH = ["Google", "Microsoft", "Amazon", "Meta", "Adobe", "Salesforce", "Oracle", "SAP"]
COMPANIES_STARTUP = ["Razorpay", "Zerodha", "Freshworks", "Postman", "Chargebee", "Hasura", "BrowserStack", "Innovaccer", "Slice", "CRED", "Groww", "Meesho"]
COMPANIES_SERVICE = ["TCS", "Infosys", "Wipro", "Cognizant", "Accenture", "Capgemini", "HCL Tech"]
COMPANIES_FINANCE = ["Goldman Sachs", "JPMorgan", "Morgan Stanley", "Deutsche Bank", "Barclays"]
ALL_COMPANIES = COMPANIES_BIGTECH + COMPANIES_STARTUP + COMPANIES_SERVICE + COMPANIES_FINANCE

TITLES_TRACK = {
    "backend": ["Software Engineer", "Senior Software Engineer", "Backend Engineer", "Staff Engineer", "SDE-2", "SDE-3", "Tech Lead"],
    "frontend": ["Frontend Engineer", "UI Engineer", "Senior Frontend Developer", "React Developer"],
    "data": ["Data Scientist", "Senior Data Scientist", "ML Engineer", "Applied Scientist", "Data Analyst", "Data Engineer"],
    "devops": ["DevOps Engineer", "SRE", "Platform Engineer", "Cloud Infrastructure Engineer"],
    "product": ["Product Manager", "Senior Product Manager", "Associate Product Manager", "Group Product Manager"],
    "design": ["Product Designer", "UX Designer", "Senior UX Researcher"],
    "qa": ["QA Engineer", "SDET", "Automation Test Engineer"],
}

SKILLS_POOL = {
    "backend": ["Python", "Java", "Go", "Node.js", "Spring Boot", "Microservices", "PostgreSQL", "MongoDB", "Redis", "Kafka", "gRPC", "REST APIs", "System Design", "AWS", "Docker", "Kubernetes"],
    "frontend": ["JavaScript", "TypeScript", "React", "Redux", "Next.js", "Vue.js", "CSS", "Webpack", "GraphQL", "Accessibility", "Performance Optimization"],
    "data": ["Python", "SQL", "Pandas", "PyTorch", "TensorFlow", "Scikit-learn", "NLP", "Computer Vision", "MLOps", "Spark", "Airflow", "A/B Testing", "Statistics", "LLMs", "Feature Engineering"],
    "devops": ["AWS", "GCP", "Azure", "Kubernetes", "Terraform", "CI/CD", "Jenkins", "Prometheus", "Grafana", "Ansible", "Linux", "Bash", "Site Reliability"],
    "product": ["Roadmapping", "Stakeholder Management", "Agile", "User Research", "SQL", "Analytics", "A/B Testing", "Wireframing", "Go-to-Market", "Pricing Strategy"],
    "design": ["Figma", "User Research", "Wireframing", "Design Systems", "Prototyping", "Usability Testing", "Adobe XD"],
    "qa": ["Selenium", "Cypress", "Pytest", "Test Automation", "API Testing", "Performance Testing", "JIRA", "CI/CD"],
}

DEGREES = ["B.Tech Computer Science", "B.E. Information Technology", "M.Tech Computer Science", "B.Sc Mathematics", "MBA", "B.Tech Electronics", "MS Computer Science", "BCA", "MCA"]
COLLEGES = ["IIT Bombay", "IIT Delhi", "IIT Madras", "BITS Pilani", "NIT Trichy", "VIT Vellore", "Anna University", "Delhi University", "IIIT Hyderabad", "Pune University", "Manipal Institute of Technology", "PES University"]

CERTS_POOL = ["AWS Certified Solutions Architect", "Google Cloud Professional ML Engineer", "PMP", "CSPO", "Azure Fundamentals", "Certified Kubernetes Administrator", "Coursera Deep Learning Specialization", None, None, None]

GITHUB_BIO_TEMPLATES = [
    "Active open-source contributor, {repos} public repos, {stars} stars total.",
    "Occasional contributor with {repos} repos, mostly personal projects.",
    "Maintains a popular {domain} library with {stars} stars.",
    "Minimal public activity, {repos} repos, mostly forks.",
]

BLOG_TOPICS = ["distributed systems", "machine learning pipelines", "career growth in tech", "system design interviews", "React performance", "leading engineering teams", "data privacy", "startup scaling lessons"]

LEADERSHIP_SNIPPETS = [
    "Led a team of {n} engineers to deliver the payments revamp.",
    "Mentored {n} junior engineers, two of whom were promoted within a year.",
    "Drove cross-functional alignment between product, design, and data teams.",
    "Owned the migration from monolith to microservices end-to-end.",
    "Initiated and ran the internal engineering guild on system design.",
    "No formal leadership experience; primarily an individual contributor.",
    "Stepped up as interim lead during a critical product launch.",
]

RED_FLAGS = [
    None, None, None, None, None,  # most candidates have none
    "Gap of 14 months between roles, unexplained in profile.",
    "Average tenure under 9 months across last 3 roles.",
    "Resume lists overlapping employment dates at two companies.",
]


def pick_track():
    return random.choice(list(TITLES_TRACK.keys()))


def gen_experience(track, years_experience):
    """Generate a plausible career history consistent with years_experience."""
    history = []
    remaining = years_experience
    current_year = 2026
    num_jobs = max(1, min(5, round(years_experience / 2.2) + random.choice([-1, 0, 0, 1])))
    num_jobs = max(1, num_jobs)

    for i in range(num_jobs):
        if remaining <= 0:
            break
        if i == num_jobs - 1:
            stint = remaining
        else:
            stint = round(random.uniform(0.8, min(3.5, remaining)), 1)
        stint = max(0.4, stint)
        end_year = current_year
        start_year = round(end_year - stint, 1)

        company_pool = random.choice([COMPANIES_BIGTECH, COMPANIES_STARTUP, COMPANIES_SERVICE, COMPANIES_FINANCE])
        company = random.choice(company_pool)
        title = random.choice(TITLES_TRACK[track])

        history.append({
            "company": company,
            "title": title,
            "start_year": start_year,
            "end_year": end_year if i > 0 else "Present",
            "duration_years": round(stint, 1),
        })
        current_year = start_year
        remaining -= stint

    history.reverse()
    return history


def gen_github_activity(seniority_score):
    """seniority_score in [0,1] biases activity upward for stronger profiles, with noise."""
    base = seniority_score * random.uniform(0.5, 1.3)
    repos = max(0, int(random.gauss(base * 25, 8)))
    stars = max(0, int(random.gauss(base * 180, 60)))
    contributions_last_year = max(0, int(random.gauss(base * 400, 150)))
    template = random.choice(GITHUB_BIO_TEMPLATES)
    domain = random.choice(["data viz", "CLI tooling", "ML", "web framework", "DevOps automation"])
    bio = template.format(repos=repos, stars=stars, domain=domain)
    return {
        "public_repos": repos,
        "github_stars_received": stars,
        "contributions_last_year": contributions_last_year,
        "github_bio": bio,
    }


def gen_blog_activity(seniority_score):
    if random.random() < 0.35 + seniority_score * 0.25:
        n = random.randint(1, 4) if seniority_score < 0.6 else random.randint(2, 9)
        topics = random.sample(BLOG_TOPICS, k=min(n, len(BLOG_TOPICS)))
        return f"Writes occasionally on: {', '.join(topics)}. {n} posts in last 2 years."
    return "No public technical writing found."


def gen_candidate(cid):
    track = pick_track()
    years_experience = round(max(0.5, random.gauss(6, 4)), 1)
    years_experience = min(years_experience, 22)

    # seniority_score is a hidden "true quality" signal used to correlate
    # related fields (so the dataset has realistic structure, not pure noise)
    seniority_score = min(1.0, max(0.05, years_experience / 14 + random.uniform(-0.15, 0.15)))

    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    experience = gen_experience(track, years_experience)
    current_title = experience[-1]["title"] if experience else random.choice(TITLES_TRACK[track])
    current_company = experience[-1]["company"] if experience else random.choice(ALL_COMPANIES)

    skill_pool = SKILLS_POOL[track]
    n_skills = random.randint(4, min(10, len(skill_pool)))
    skills = random.sample(skill_pool, k=n_skills)
    # occasionally add a cross-track skill for realism (T-shaped folks)
    if random.random() < 0.3:
        other_track = random.choice([t for t in SKILLS_POOL if t != track])
        skills.append(random.choice(SKILLS_POOL[other_track]))

    education = {
        "degree": random.choice(DEGREES),
        "college": random.choice(COLLEGES),
        "grad_year": int(2026 - years_experience - random.uniform(0, 1.5)),
    }

    leadership = random.choice(LEADERSHIP_SNIPPETS)
    if seniority_score > 0.55 and "No formal leadership" in leadership:
        leadership = random.choice([l for l in LEADERSHIP_SNIPPETS if "No formal" not in l])
    leadership = leadership.format(n=random.randint(2, 8)) if "{n}" in leadership else leadership

    cert = random.choice(CERTS_POOL)
    github = gen_github_activity(seniority_score)
    blog = gen_blog_activity(seniority_score)
    red_flag = random.choice(RED_FLAGS)

    notice_period_days = random.choice([0, 15, 30, 30, 30, 60, 90])
    expected_salary_lpa = round(max(4, years_experience * random.uniform(2.2, 4.0) + random.uniform(-3, 3)), 1)
    location = random.choice(["Bangalore", "Hyderabad", "Pune", "Mumbai", "Delhi NCR", "Chennai", "Remote", "Bangalore", "Bangalore"])

    summary = (
        f"{current_title} with {years_experience} years of experience, currently at {current_company}. "
        f"Core strengths in {', '.join(skills[:3])}. {leadership}"
    )

    return {
        "candidate_id": f"CAND{cid:04d}",
        "name": name,
        "track": track,
        "current_title": current_title,
        "current_company": current_company,
        "years_experience": years_experience,
        "location": location,
        "skills": "; ".join(skills),
        "education_degree": education["degree"],
        "education_college": education["college"],
        "grad_year": education["grad_year"],
        "certifications": cert or "",
        "career_history": json.dumps(experience),
        "leadership_notes": leadership,
        "github_public_repos": github["public_repos"],
        "github_stars_received": github["github_stars_received"],
        "github_contributions_last_year": github["contributions_last_year"],
        "github_bio": github["github_bio"],
        "blog_activity": blog,
        "behavioral_flags": red_flag or "",
        "notice_period_days": notice_period_days,
        "expected_salary_lpa": expected_salary_lpa,
        "profile_summary": summary,
        "_true_seniority_score": round(seniority_score, 3),  # hidden ground-truth, for eval only
    }


def main():
    n_candidates = 120
    candidates = [gen_candidate(i + 1) for i in range(n_candidates)]

    # ensure track diversity is reasonable: bias dataset toward backend/data/devops
    # since our sample JD will target a backend/platform role
    fieldnames = [k for k in candidates[0].keys()]

    out_csv = OUT_DIR / "candidates.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            writer.writerow(c)

    jd_text = """Senior Backend Engineer — Payments Platform

About the Role:
We're hiring a Senior Backend Engineer to join our Payments Platform team. You'll design and
build high-throughput, low-latency services that process financial transactions at scale, and
you'll be a technical anchor for a small, fast-moving team.

What you'll do:
- Design and own backend services handling payment processing, reconciliation, and ledger systems
- Build for reliability: our systems need to be correct under failure, not just fast on the happy path
- Mentor 1-2 mid-level engineers and raise the team's engineering bar through code review and design docs
- Partner closely with product and risk teams to translate ambiguous requirements into robust systems
- Debug production incidents under pressure and drive postmortems that prevent recurrence

What we're looking for:
- 5+ years of backend engineering experience, ideally with distributed systems or financial/payments systems
- Strong CS fundamentals: system design, data structures, concurrency, database internals
- Hands-on experience with at least one of: Python, Java, or Go, plus SQL and a messaging system (Kafka or similar)
- Experience operating services in production on AWS or GCP, comfort with Docker/Kubernetes
- A track record of mentoring or technical leadership — we want someone who can grow with the team, not just write code in isolation
- Evidence of intellectual curiosity: open-source contributions, technical writing, or a history of going deep on hard problems
- Comfortable with ambiguity and able to push back constructively on requirements that don't make sense

Nice to have:
- Experience in fintech, payments, or other domains with strict correctness/compliance requirements
- Familiarity with ML-driven fraud detection or risk scoring systems
- Startup experience where you owned a system end-to-end with limited oversight

We care less about pedigree and more about whether you've actually built and operated systems
that mattered. A candidate who shipped something real at a small startup is just as compelling to
us as one from a brand-name company — we're looking for substance over keywords.
"""
    out_jd = OUT_DIR / "job_description.txt"
    out_jd.write_text(jd_text, encoding="utf-8")

    print(f"Generated {len(candidates)} candidates -> {out_csv}")
    print(f"Generated job description -> {out_jd}")


if __name__ == "__main__":
    main()
