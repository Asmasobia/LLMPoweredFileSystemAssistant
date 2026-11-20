# Eval review sheet

Two parts. Type only in the **your fix** column and on the **Grade:** lines,
then tell Claude, who runs `python evals/review_sheet.py apply`.

## Part 1: retrieval labels (53 queries)

Ask: *if a recruiter typed this, which résumés would they want to see?*
Leave **your fix** blank if the list is right. Otherwise `+name` to add, `-name` to remove
(names as in `resumes/`, without `resume_` and `.txt`). The ones marked CHECK are judgement calls.
Résumé names: alex_kumar, amanda_davis, andrew_young, brian_jackson, chris_lee, christopher_lee2, daniel_johnson, derek_campbell, hannah_murphy, james_adams, jane_smith, jason_wright, jennifer_white, john_doe, kevin_thomas, laura_hall, marcus_cooper, maria_garcia, megan_green, michelle_king, nathan_patel, nicole_harris, olivia_baker, rachel_moore, robert_martinez, ryan_walker, samantha_nelson, stephanie_clark, tyler_scott, victoria_ross, william_carter

**I have checked every row in Part 1:** no   ← change to `yes` when done (until then no label counts as reviewed)

| id | query | filter | relevant | note | your fix |
|---|---|---|---|---|---|
| k09 | gRPC microservices in Go |  | christopher_lee2 | CHECK: nathan_patel also lists gRPC, but works in Python; left out because the query asks for Go |  |
| s02 | builds dashboards for executives |  | hannah_murphy, olivia_baker | CHECK: both build executive dashboards; jane_smith and robert_martinez also made Tableau dashboards, left out as not their focus |  |
| s12 | deploying and monitoring machine learning models in production |  | marcus_cooper | CHECK: MLOps engineer; victoria_ross deployed models to edge devices, left out |  |
| s20 | batch ETL jobs and data warehousing |  | robert_martinez, samantha_nelson, olivia_baker | CHECK: robert and samantha are data engineers; olivia_baker 'created ETL processes' as a BI analyst |  |
| f03 | machine learning engineer | ≥4 yrs | megan_green, jane_smith | CHECK: excludes rachel/marcus/victoria (3 yrs) and jennifer_white (PM) |  |
| f04 | React developer | ≥4 yrs | ryan_walker, john_doe | CHECK: hannah_murphy (4) built React dashboards, chris_lee (4) lists React Native; both left out |  |
| k01 | Solidity smart contracts |  | amanda_davis | only Solidity mention |  |
| k02 | KiCad PCB design |  | jason_wright | only KiCad mention |  |
| k03 | Istio service mesh |  | tyler_scott | only Istio mention |  |
| k04 | Apex and Lightning Web Components |  | derek_campbell | only Salesforce résumé |  |
| k05 | Selenium test automation |  | daniel_johnson | only Selenium mention |  |
| k06 | Unity game development in C# |  | nicole_harris | only Unity / C# résumé |  |
| k07 | Burp Suite penetration testing |  | laura_hall | only security résumé |  |
| k08 | dbt models for data transformation |  | samantha_nelson | only dbt mention |  |
| k10 | MQTT and BLE IoT firmware |  | jason_wright | only MQTT/BLE mention |  |
| k11 | Kubeflow ML pipelines |  | marcus_cooper | only Kubeflow mention |  |
| k12 | SwiftUI iOS apps |  | brian_jackson, chris_lee | both build SwiftUI apps |  |
| k13 | Jetpack Compose Android |  | chris_lee | only Jetpack mention |  |
| k14 | React Native developer |  | james_adams, chris_lee | james_adams is a React Native developer; chris_lee lists React Native among frameworks |  |
| k15 | Docusaurus documentation sites |  | michelle_king | only Docusaurus mention |  |
| k16 | Snowflake |  | robert_martinez, hannah_murphy, olivia_baker | all three list Snowflake; robert_martinez managed warehouses on it |  |
| k17 | Prometheus and Grafana monitoring |  | kevin_thomas, maria_garcia, tyler_scott, marcus_cooper | all four set up Prometheus + Grafana |  |
| k18 | Terraform and Ansible automation |  | maria_garcia, kevin_thomas | only two with both |  |
| k19 | Kafka event streaming |  | andrew_young, christopher_lee2, robert_martinez, samantha_nelson | every Kafka mention |  |
| k20 | OSCP certified |  | laura_hall | only OSCP mention |  |
| k21 | AWS Solutions Architect Professional certification |  | andrew_young, maria_garcia | both hold it |  |
| k22 | OAuth 2.0 and JWT authentication |  | alex_kumar | only OAuth/JWT mention |  |
| s01 | someone who keeps production systems up and handles on-call incidents |  | kevin_thomas | SRE: uptime, incident response, PagerDuty |  |
| s03 | writer for developer docs and API references |  | michelle_king | technical writer |  |
| s04 | decentralized finance on the blockchain |  | amanda_davis | DeFi protocols |  |
| s05 | person who finds vulnerabilities by attacking our web application |  | laura_hall | penetration testing |  |
| s06 | low-level programmer for microcontrollers |  | jason_wright | firmware for ARM Cortex; william_carter is low-level but not microcontrollers |  |
| s07 | memory-safe high-performance networking code |  | william_carter | Rust networking libraries |  |
| s08 | perception models for self-driving cars |  | victoria_ross, megan_green | victoria: autonomous vehicles; megan: CV for autonomous systems |  |
| s09 | teaching computers to understand text and build chatbots |  | rachel_moore | NLP engineer |  |
| s10 | keeps databases replicated and backed up |  | stephanie_clark | DBA: replication, failover, backup |  |
| s11 | customizes CRM software for enterprise sales teams |  | derek_campbell | Salesforce developer |  |
| s13 | owns the product roadmap and works with engineers |  | jennifer_white | technical product manager |  |
| s14 | builds games with physics and shaders |  | nicole_harris | game developer |  |
| s15 | one codebase for both iPhone and Android apps |  | chris_lee, james_adams | Flutter / React Native cross-platform |  |
| s16 | checks that our APIs work and holds up under load |  | daniel_johnson | QA: API testing, JMeter load testing |  |
| s17 | break a monolith into services in the cloud |  | andrew_young | led monolith-to-microservices migration on AWS |  |
| s18 | designs public APIs with rate limiting and versioning |  | nathan_patel | API developer |  |
| s19 | real-time streaming data pipelines |  | samantha_nelson | Kafka + Spark Streaming + Flink; robert_martinez is batch ETL |  |
| s21 | predicting which customers will leave |  | jane_smith | churn prediction |  |
| s22 | generating synthetic images with neural networks |  | megan_green | GANs for image synthesis |  |
| s23 | builds tooling so other engineering teams can ship on their own |  | tyler_scott | internal developer platform, self-service |  |
| s24 | full stack developer with a Python backend and React frontend |  | ryan_walker, john_doe | alex_kumar is full stack but JavaScript, not Python |  |
| s25 | adapting large language models to our domain |  | rachel_moore | fine-tuned LLMs |  |
| f01 | Kafka | ≥5 yrs | andrew_young | Kafka: andrew 8, christopher_lee2 4, robert 4, samantha 4 |  |
| f02 | Kubernetes | ≥6 yrs | andrew_young, maria_garcia | Kubernetes with 6+ years: andrew 8, maria 6 |  |
| f05 | cloud infrastructure on AWS | ≥6 yrs | andrew_young, maria_garcia | stephanie_clark (6) uses AWS RDS but is a DBA |  |
| f06 | test automation in CI/CD | ≥5 yrs | daniel_johnson | QA, 5 yrs |  |

## Part 2: answer grading (grade at least 20 of 24)

For each answer: is **every claim it makes about a candidate** stated in the excerpts below it?
Judge against those excerpts only, not the full résumés. "The résumés don't say" counts as grounded.
Write `g` (grounded) or `u` (ungrounded) after **Grade:**; optionally ` - reason`.

---

### g01: Which candidates have worked with Kafka, and in what role?

**Answer:**

> **Candidates who have worked with Kafka and their roles**
> 
> | Candidate | Role in which Kafka was used | Source |
> |-----------|------------------------------|--------|
> | Andrew Young | Senior Solutions Architect (TechBridge) – “Architected event‑driven systems using Kafka and AWS Lambda” | resume_andrew_young.txt |
> | Christopher Lee | Backend Developer (ServerTech) – “Built event‑driven systems with Kafka” | resume_christopher_lee2.txt |
> | Samantha Nelson | Senior Data Engineer (CloudData Inc.) – “Built real‑time data pipelines using Kafka and Spark Streaming” | resume_samantha_nelson.txt |
> 
> *The résumé excerpts for Robert Martinez do not mention any role in which Kafka was used.*

<details><summary>Excerpts the model was given</summary>

```
[1] resume_andrew_young.txt (EXPERIENCE)
Principal Solutions Architect | EnterpriseCloud | 2020 - Present
- Designed cloud-native architectures for Fortune 500 clients
- Led migration of monolithic systems to microservices on AWS
- Reduced infrastructure costs by 45% through optimization

Senior Solutions Architect | TechBridge | 2016 - 2020
- Architected event-driven systems using Kafka and AWS Lambda
- Designed multi-tenant SaaS platforms
- Mentored junior architects and engineers

[2] resume_christopher_lee2.txt (EXPERIENCE)
Senior Backend Developer | GoScale Inc. | 2022 - Present
- Built high-performance microservices in Go handling 100K+ RPS
- Designed gRPC APIs and protocol buffer schemas
- Implemented distributed tracing with Jaeger

Backend Developer | ServerTech | 2020 - 2022
- Developed REST APIs using Go and Python Flask
- Managed PostgreSQL and Redis databases
- Built event-driven systems with Kafka

[3] resume_samantha_nelson.txt (EXPERIENCE)
Senior Data Engineer | CloudData Inc. | 2022 - Present
- Built real-time data pipelines using Kafka and Spark Streaming
- Designed data lake architecture on AWS S3 and Glue
- Implemented data governance and quality frameworks

Data Engineer | DataStream | 2020 - 2022
- Developed batch ETL jobs using Python and Airflow
- Managed data warehouses on BigQuery and Redshift
- Created dbt models for data transformation

[4] resume_robert_martinez.txt (SKILLS)
Languages: Python, SQL, Scala
Tools: Spark, Airflow, Snowflake, Redshift, Kafka
Databases: PostgreSQL, MongoDB, BigQuery
Cloud: AWS, GCP

[5] resume_andrew_young.txt (SKILLS)
Cloud: AWS, Azure, GCP
Languages: Python, Java, TypeScript, Go
Architecture: Microservices, Event-Driven, Serverless, CQRS
Tools: Terraform, Docker, Kubernetes, Kafka, Redis
```

</details>

**Grade:** 

---

### g02: How many years of experience does Maria Garcia have, and which cloud certifications does she hold?

**Answer:**

> Maria Garcia has **6 years of experience** (resume_maria_garcia.txt) and holds **AWS and Azure certifications** (resume_maria_garcia.txt).

<details><summary>Excerpts the model was given</summary>

```
[1] resume_maria_garcia.txt (SUMMARY)
DevOps engineer with 6 years of experience in cloud infrastructure,
CI/CD pipelines, and container orchestration. AWS and Azure certified.

[2] resume_samantha_nelson.txt (SUMMARY)
Cloud data engineer with 4 years of experience building data platforms on AWS and GCP.

[3] resume_maria_garcia.txt (HEADER)
MARIA GARCIA
DevOps Engineer

[4] resume_amanda_davis.txt (SUMMARY)
Blockchain developer with 3 years of experience building decentralized applications
and smart contracts on Ethereum and Solana.

[5] resume_brian_jackson.txt (SUMMARY)
iOS developer with 4 years of experience building native Apple applications.
```

</details>

**Grade:** 

---

### g03: Who has built iOS apps with SwiftUI?

**Answer:**

> Brian Jackson and Chris Lee have both built iOS apps with SwiftUI.  
> - Brian Jackson: “Built iOS apps using Swift and SwiftUI” (resume_brian_jackson.txt).  
> - Chris Lee: “Developed iOS apps using Swift and SwiftUI” (resume_chris_lee.txt).

<details><summary>Excerpts the model was given</summary>

```
[1] resume_brian_jackson.txt (EXPERIENCE)
Senior iOS Developer | AppsForward | 2022 - Present
- Built iOS apps using Swift and SwiftUI
- Implemented Core Data and CloudKit integration
- Published 3 apps with 100K+ downloads

iOS Developer | MobileMinds | 2020 - 2022
- Developed UIKit-based applications
- Integrated RESTful APIs and Firebase
- Implemented in-app purchases and subscriptions

[2] resume_chris_lee.txt (EXPERIENCE)
Senior Mobile Developer | AppWorks Studio | 2022 - Present
- Developed iOS apps using Swift and SwiftUI
- Built cross-platform apps with Flutter and Dart
- Integrated Firebase for authentication and real-time databases

Mobile Developer | MobileFirst Inc. | 2020 - 2022
- Created Android apps using Kotlin and Jetpack Compose
- Implemented push notifications and in-app purchases
- Published 5 apps on App Store and Google Play

[3] resume_james_adams.txt (EXPERIENCE)
React Native Developer | MobileApps Co. | 2022 - Present
- Built cross-platform mobile apps using React Native and Expo
- Integrated native modules for iOS and Android
- Implemented state management with Redux and MobX

Junior Mobile Developer | AppStudio | 2021 - 2022
- Developed UI components using React Native
- Integrated REST APIs and Firebase
- Published 4 apps on App Store and Google Play

[4] resume_chris_lee.txt (SUMMARY)
Mobile developer with 4 years of experience building iOS and Android apps.
Proficient in Swift, Kotlin, and cross-platform development with Flutter.

[5] resume_brian_jackson.txt (SUMMARY)
iOS developer with 4 years of experience building native Apple applications.
```

</details>

**Grade:** 

---

### g04: What security certifications does Laura Hall hold?

**Answer:**

> Laura Hall holds the following security certifications: CompTIA Security+ and OSCP (Offensive Security Certified Professional)【resume_laura_hall.txt】.

<details><summary>Excerpts the model was given</summary>

```
[1] resume_laura_hall.txt (HEADER)
LAURA HALL
Security Engineer

[2] resume_laura_hall.txt (SUMMARY)
Security engineer with 4 years of experience in application security and penetration testing.

[3] resume_laura_hall.txt (EXPERIENCE)
Security Engineer | AppSec Corp | 2022 - Present
- Conducted application security assessments and code reviews
- Implemented SAST/DAST tools in CI/CD pipelines
- Developed security automation using Python

Security Analyst | SecureApps | 2020 - 2022
- Performed penetration testing on web applications
- Managed vulnerability scanning with Nessus and Qualys
- Created security training materials for developers

[4] resume_laura_hall.txt (EDUCATION)
M.S. Cybersecurity | Johns Hopkins University | 2020
B.S. Computer Science | University of Maryland | 2018
CompTIA Security+, OSCP

[5] resume_maria_garcia.txt (EDUCATION)
B.S. Information Technology | University of Washington | 2018
AWS Solutions Architect Professional Certification
```

</details>

**Grade:** 

---

### g05: Which candidate has published research papers, and on what topics?

**Answer:**

> The résumé excerpts indicate that **Megan Green** has published research papers.  
> - She has published **3 papers on object detection and segmentation** (resume_megan_green.txt, EXPERIENCE section).  
> 
> No other candidate’s résumé excerpts mention any published research papers.

<details><summary>Excerpts the model was given</summary>

```
[1] resume_megan_green.txt (HEADER)
MEGAN GREEN
AI Research Engineer

[2] resume_jennifer_white.txt (EXPERIENCE)
Senior Product Manager | ProductHub | 2021 - Present
- Led roadmap for ML-powered recommendation engine
- Managed team of 8 engineers and 2 designers
- Increased user retention by 25% through feature optimization

Product Manager | TechStartup | 2018 - 2021
- Defined product requirements for SaaS platform
- Conducted user research and A/B testing
- Worked closely with Python and React engineering teams

[3] resume_megan_green.txt (EXPERIENCE)
AI Research Engineer | VisionAI Labs | 2022 - Present
- Developed computer vision models for autonomous systems using PyTorch
- Published 3 papers on object detection and segmentation
- Built data annotation pipelines processing 1M+ images

ML Research Engineer | DeepTech | 2020 - 2022
- Trained GANs for image synthesis and augmentation
- Implemented transformer architectures for vision tasks
- Optimized model inference using TensorRT and ONNX

[4] resume_megan_green.txt (SUMMARY)
AI research engineer with 4 years of experience in deep learning and computer vision.

[5] resume_ryan_walker.txt (EDUCATION)
B.S. Computer Science | Vanderbilt University | 2020
```

</details>

**Grade:** 

---

### g06: What is Andrew Young's highest degree?

**Answer:**

> The résumé excerpts provided do not say what Andrew Young’s highest degree is.

<details><summary>Excerpts the model was given</summary>

```
[1] resume_andrew_young.txt (HEADER)
ANDREW YOUNG
Solutions Architect

[2] resume_john_doe.txt (HEADER)
JOHN DOE
Software Engineer

[3] resume_marcus_cooper.txt (HEADER)
MARCUS COOPER
MLOps Engineer

[4] resume_tyler_scott.txt (HEADER)
TYLER SCOTT
Platform Engineer

[5] resume_jennifer_white.txt (EDUCATION)
M.B.A. | Stanford University | 2018
B.S. Computer Science | UC San Diego | 2015
```

</details>

**Grade:** 

---

### g07: Who could lead a migration from on-premise servers to AWS?

**Answer:**

> The résumé excerpts indicate that **Maria Garcia** has experience leading a migration from on‑premise servers to AWS. She worked as a DevOps Engineer at InfraTech, where she “Migrated on‑premise applications to AWS” (resume_maria_garcia.txt).

<details><summary>Excerpts the model was given</summary>

```
[1] resume_andrew_young.txt (EXPERIENCE)
Principal Solutions Architect | EnterpriseCloud | 2020 - Present
- Designed cloud-native architectures for Fortune 500 clients
- Led migration of monolithic systems to microservices on AWS
- Reduced infrastructure costs by 45% through optimization

Senior Solutions Architect | TechBridge | 2016 - 2020
- Architected event-driven systems using Kafka and AWS Lambda
- Designed multi-tenant SaaS platforms
- Mentored junior architects and engineers

[2] resume_maria_garcia.txt (EXPERIENCE)
Senior DevOps Engineer | CloudScale Inc. | 2021 - Present
- Managed Kubernetes clusters serving 10M+ requests daily
- Automated infrastructure using Terraform and Ansible
- Reduced deployment time by 70% with optimized CI/CD pipelines

DevOps Engineer | InfraTech | 2018 - 2021
- Set up monitoring with Prometheus, Grafana, and ELK stack
- Migrated on-premise applications to AWS
- Implemented Docker containerization for 20+ services

[3] resume_andrew_young.txt (EDUCATION)
M.S. Software Engineering | University of North Carolina | 2016
B.S. Computer Science | NC State University | 2014
AWS Solutions Architect Professional

[4] resume_andrew_young.txt (SKILLS)
Cloud: AWS, Azure, GCP
Languages: Python, Java, TypeScript, Go
Architecture: Microservices, Event-Driven, Serverless, CQRS
Tools: Terraform, Docker, Kubernetes, Kafka, Redis

[5] resume_john_doe.txt (EXPERIENCE)
Senior Software Engineer | TechCorp Inc. | 2021 - Present
- Developed microservices using Python and FastAPI
- Managed AWS infrastructure with Terraform
- Led a team of 4 engineers

Software Engineer | StartupXYZ | 2019 - 2021
- Built full-stack applications with React and Node.js
- Implemented CI/CD pipelines using GitHub Actions
- Worked with PostgreSQL and MongoDB databases
```

</details>

**Grade:** 

---

### g08: Which candidates know both Python and Go?

**Answer:**

> Candidates who list both Python and Go in their résumés are:
> 
> - Christopher Lee – languages include Go and Python (resume_christopher_lee2.txt).  
> - Marcus Cooper – languages include Python and Go (resume_marcus_cooper.txt).

<details><summary>Excerpts the model was given</summary>

```
[1] resume_christopher_lee2.txt (SUMMARY)
Backend developer with 4 years of experience building microservices in Go and Python.

[2] resume_ryan_walker.txt (EXPERIENCE)
Full Stack Developer | PyWeb Solutions | 2022 - Present
- Built RESTful APIs using Django and Django REST Framework
- Developed frontend using React and TypeScript
- Deployed applications on AWS using Docker and ECS

Python Developer | CodeShop | 2020 - 2022
- Created web scrapers and automation tools in Python
- Built Flask APIs with SQLAlchemy ORM
- Implemented Celery task queues with Redis

[3] resume_christopher_lee2.txt (SKILLS)
Languages: Go, Python, SQL, TypeScript
Frameworks: Gin, Echo, Flask, FastAPI
Databases: PostgreSQL, Redis, Cassandra, MongoDB
Tools: Docker, Kubernetes, Kafka, gRPC, Jaeger

[4] resume_marcus_cooper.txt (SKILLS)
Languages: Python, Bash, Go
MLOps: MLflow, Kubeflow, Feast, DVC, Weights & Biases
Infrastructure: Docker, Kubernetes, Terraform, ArgoCD
Cloud: AWS SageMaker, GCP Vertex AI, Azure ML

[5] resume_jane_smith.txt (SUMMARY)
Data scientist with 4 years of experience in machine learning, statistical analysis,
and data visualization. Skilled in Python and R.
```

</details>

**Grade:** 

---

### g09: What salary does Kevin Thomas expect?

**Answer:**

> The résumés provided do not say.

<details><summary>Excerpts the model was given</summary>

```
[1] resume_kevin_thomas.txt (HEADER)
KEVIN THOMAS
Site Reliability Engineer

[2] resume_kevin_thomas.txt (CONTACT)
Email: kevin.thomas@email.com
Phone: (555) 555-6677
Location: Minneapolis, MN

[3] resume_marcus_cooper.txt (HEADER)
MARCUS COOPER
MLOps Engineer

[4] resume_daniel_johnson.txt (HEADER)
DANIEL JOHNSON
QA Automation Engineer

[5] resume_christopher_lee2.txt (HEADER)
CHRISTOPHER LEE
Go Backend Developer
```

</details>

**Grade:** 

---

### g10: Is Jennifer White willing to relocate?

**Answer:**

> The résumés provided do not say.

<details><summary>Excerpts the model was given</summary>

```
[1] resume_jennifer_white.txt (HEADER)
JENNIFER WHITE
Product Manager (Technical)

[2] resume_jennifer_white.txt (CONTACT)
Email: jennifer.white@email.com
Phone: (555) 222-3344
Location: San Jose, CA

[3] resume_maria_garcia.txt (HEADER)
MARIA GARCIA
DevOps Engineer

[4] resume_amanda_davis.txt (CONTACT)
Email: amanda.davis@email.com
Phone: (555) 444-5566
Location: Atlanta, GA

[5] resume_rachel_moore.txt (CONTACT)
Email: rachel.moore@email.com
Phone: (555) 666-7788
Location: Pittsburgh, PA
```

</details>

**Grade:** 

---

### g11: Which candidates speak Spanish?

**Answer:**

> The résumés provided do not say that any of the candidates speak Spanish.

<details><summary>Excerpts the model was given</summary>

```
[1] resume_rachel_moore.txt (EDUCATION)
M.S. Computational Linguistics | Carnegie Mellon University | 2021
B.A. Linguistics | Penn State University | 2019

[2] resume_maria_garcia.txt (HEADER)
MARIA GARCIA
DevOps Engineer

[3] resume_jennifer_white.txt (EDUCATION)
M.B.A. | Stanford University | 2018
B.S. Computer Science | UC San Diego | 2015

[4] resume_robert_martinez.txt (EDUCATION)
B.S. Computer Science | University of Miami | 2020

[5] resume_rachel_moore.txt (EXPERIENCE)
NLP Engineer | LangAI Inc. | 2022 - Present
- Fine-tuned large language models using Hugging Face Transformers
- Built chatbot systems with RAG architecture
- Developed text classification pipelines with 95% accuracy

Junior ML Engineer | TextMine | 2021 - 2022
- Preprocessed text data using spaCy and NLTK
- Trained sentiment analysis models using BERT
- Created named entity recognition systems
```

</details>

**Grade:** 

---

### g12: What was Derek Campbell's GPA?

**Answer:**

> The résumés provided do not say.

<details><summary>Excerpts the model was given</summary>

```
[1] resume_derek_campbell.txt (HEADER)
DEREK CAMPBELL
Salesforce Developer

[2] resume_john_doe.txt (HEADER)
JOHN DOE
Software Engineer

[3] resume_alex_kumar.txt (EDUCATION)
B.S. Computer Science | University of Texas at Austin | 2021

[4] resume_tyler_scott.txt (HEADER)
TYLER SCOTT
Platform Engineer

[5] resume_jennifer_white.txt (EDUCATION)
M.B.A. | Stanford University | 2018
B.S. Computer Science | UC San Diego | 2015
```

</details>

**Grade:** 

---

### g13: Who has experience with real-time data pipelines?

**Answer:**

> The résumé that explicitly mentions building real‑time data pipelines is **Samantha Nelson** (resume_samantha_nelson.txt).

<details><summary>Excerpts the model was given</summary>

```
[1] resume_samantha_nelson.txt (EXPERIENCE)
Senior Data Engineer | CloudData Inc. | 2022 - Present
- Built real-time data pipelines using Kafka and Spark Streaming
- Designed data lake architecture on AWS S3 and Glue
- Implemented data governance and quality frameworks

Data Engineer | DataStream | 2020 - 2022
- Developed batch ETL jobs using Python and Airflow
- Managed data warehouses on BigQuery and Redshift
- Created dbt models for data transformation

[2] resume_robert_martinez.txt (EXPERIENCE)
Data Engineer | DataFlow Inc. | 2021 - Present
- Built ETL pipelines using Apache Spark and Airflow
- Managed data warehouses on Snowflake and Redshift
- Optimized SQL queries reducing runtime by 50%

Junior Data Engineer | InfoTech | 2020 - 2021
- Developed Python scripts for data cleaning and transformation
- Created dashboards using Tableau
- Worked with PostgreSQL and MongoDB

[3] resume_hannah_murphy.txt (EXPERIENCE)
Senior Data Viz Engineer | VizWorks | 2022 - Present
- Built interactive dashboards using D3.js and React
- Created real-time data visualizations using WebSockets
- Designed Tableau dashboards for executive reporting

Data Analyst | ChartSmart Inc. | 2020 - 2022
- Developed Python data pipelines using Pandas and NumPy
- Created automated reports using Matplotlib and Seaborn
- Built Power BI dashboards tracking KPIs for 10+ teams

[4] resume_robert_martinez.txt (SUMMARY)
Data engineer with 4 years of experience building ETL pipelines and data warehouses.

[5] resume_jane_smith.txt (EXPERIENCE)
Senior Data Scientist | DataDriven Corp | 2022 - Present
- Built ML models for customer churn prediction with 92% accuracy
- Created dashboards using Tableau and Power BI
- Led A/B testing initiatives across product teams

Data Analyst | Analytics Pro | 2020 - 2022
- Analyzed large datasets using Python and SQL
- Developed automated reporting pipelines
- Collaborated with engineering teams on data infrastructure
```

</details>

**Grade:** 

---

### g14: Which candidate has built a chatbot using RAG?

**Answer:**

> The candidate who has built a chatbot using RAG is **Rachel Moore** (resume_rachel_moore.txt).

<details><summary>Excerpts the model was given</summary>

```
[1] resume_rachel_moore.txt (EXPERIENCE)
NLP Engineer | LangAI Inc. | 2022 - Present
- Fine-tuned large language models using Hugging Face Transformers
- Built chatbot systems with RAG architecture
- Developed text classification pipelines with 95% accuracy

Junior ML Engineer | TextMine | 2021 - 2022
- Preprocessed text data using spaCy and NLTK
- Trained sentiment analysis models using BERT
- Created named entity recognition systems

[2] resume_alex_kumar.txt (EXPERIENCE)
Full Stack Developer | WebSolutions Inc. | 2022 - Present
- Built responsive web apps using React and TypeScript
- Designed RESTful APIs with Node.js and Express
- Implemented authentication using OAuth 2.0 and JWT

Junior Developer | CodeStart | 2021 - 2022
- Developed frontend components using HTML, CSS, JavaScript
- Wrote unit tests with Jest and React Testing Library
- Participated in daily standups and sprint planning

[3] resume_hannah_murphy.txt (EXPERIENCE)
Senior Data Viz Engineer | VizWorks | 2022 - Present
- Built interactive dashboards using D3.js and React
- Created real-time data visualizations using WebSockets
- Designed Tableau dashboards for executive reporting

Data Analyst | ChartSmart Inc. | 2020 - 2022
- Developed Python data pipelines using Pandas and NumPy
- Created automated reports using Matplotlib and Seaborn
- Built Power BI dashboards tracking KPIs for 10+ teams

[4] resume_derek_campbell.txt (EXPERIENCE)
Senior Salesforce Developer | CRMPro Solutions | 2021 - Present
- Developed custom Salesforce applications using Apex and Lightning Web Components
- Built integrations with external systems using REST APIs and MuleSoft
- Managed data migrations for 500K+ records using Python and Data Loader

Salesforce Developer | CloudCRM Inc. | 2019 - 2021
- Customized Salesforce workflows and process builders
- Created Visualforce pages and Lightning components
- Wrote unit tests achieving 90%+ code coverage

[5] resume_rachel_moore.txt (SKILLS)
Languages: Python, SQL
Libraries: Hugging Face, spaCy, NLTK, PyTorch, TensorFlow
Tools: Docker, MLflow, Weights & Biases
Specialties: NLP, LLMs, RAG, Text Classification, Chatbots
```

</details>

**Grade:** 

---

### g15: Which database systems has Stephanie Clark administered?

**Answer:**

> Stephanie Clark has administered the following database systems:
> 
> - PostgreSQL (Senior DBA, DataCore Systems)【resume_stephanie_clark.txt (EXPERIENCE)】  
> - MySQL (Senior DBA, DataCore Systems)【resume_stephanie_clark.txt (EXPERIENCE)】  
> - Oracle (DBA, InfoStore Inc.)【resume_stephanie_clark.txt (EXPERIENCE)】  
> - SQL Server (DBA, InfoStore Inc.)【resume_stephanie_clark.txt (EXPERIENCE)】

<details><summary>Excerpts the model was given</summary>

```
[1] resume_stephanie_clark.txt (HEADER)
STEPHANIE CLARK
Database Administrator

[2] resume_stephanie_clark.txt (EXPERIENCE)
Senior DBA | DataCore Systems | 2021 - Present
- Managed PostgreSQL and MySQL clusters with 99.9% uptime
- Implemented database replication and failover strategies
- Optimized query performance for high-traffic applications

DBA | InfoStore Inc. | 2018 - 2021
- Administered Oracle and SQL Server databases
- Created backup and disaster recovery procedures
- Automated routine maintenance with Python scripts

[3] resume_stephanie_clark.txt (SUMMARY)
DBA with 6 years of experience managing enterprise database systems.

[4] resume_daniel_johnson.txt (EDUCATION)
B.S. Information Systems | Arizona State University | 2019

[5] resume_derek_campbell.txt (EDUCATION)
B.S. Information Systems | Indiana University | 2019
```

</details>

**Grade:** 

---

### g16: Which candidates have a PhD?

**Answer:**

> The only candidate listed with a Ph.D. is:
> 
> - **Megan Green** – Ph.D. in Computer Science (AI) from Stanford University (2020) (resume_megan_green.txt).

<details><summary>Excerpts the model was given</summary>

```
[1] resume_megan_green.txt (EDUCATION)
Ph.D. Computer Science (AI) | Stanford University | 2020
B.S. Computer Science | UC Berkeley | 2016

[2] resume_ryan_walker.txt (EDUCATION)
B.S. Computer Science | Vanderbilt University | 2020

[3] resume_jennifer_white.txt (EDUCATION)
M.B.A. | Stanford University | 2018
B.S. Computer Science | UC San Diego | 2015

[4] resume_jane_smith.txt (EDUCATION)
M.S. Data Science | Columbia University | 2020
B.S. Statistics | University of Michigan | 2018

[5] resume_nathan_patel.txt (EDUCATION)
B.S. Computer Science | UC San Diego | 2020
```

</details>

**Grade:** 

---

### g17: Who has worked on autonomous vehicles?

**Answer:**

> Victoria Ross has worked on autonomous vehicles (resume_victoria_ross.txt).

<details><summary>Excerpts the model was given</summary>

```
[1] resume_victoria_ross.txt (SUMMARY)
Computer vision engineer with 3 years of experience building image recognition
and video analysis systems for autonomous vehicles.

[2] resume_megan_green.txt (EXPERIENCE)
AI Research Engineer | VisionAI Labs | 2022 - Present
- Developed computer vision models for autonomous systems using PyTorch
- Published 3 papers on object detection and segmentation
- Built data annotation pipelines processing 1M+ images

ML Research Engineer | DeepTech | 2020 - 2022
- Trained GANs for image synthesis and augmentation
- Implemented transformer architectures for vision tasks
- Optimized model inference using TensorRT and ONNX

[3] resume_john_doe.txt (EXPERIENCE)
Senior Software Engineer | TechCorp Inc. | 2021 - Present
- Developed microservices using Python and FastAPI
- Managed AWS infrastructure with Terraform
- Led a team of 4 engineers

Software Engineer | StartupXYZ | 2019 - 2021
- Built full-stack applications with React and Node.js
- Implemented CI/CD pipelines using GitHub Actions
- Worked with PostgreSQL and MongoDB databases

[4] resume_jennifer_white.txt (EXPERIENCE)
Senior Product Manager | ProductHub | 2021 - Present
- Led roadmap for ML-powered recommendation engine
- Managed team of 8 engineers and 2 designers
- Increased user retention by 25% through feature optimization

Product Manager | TechStartup | 2018 - 2021
- Defined product requirements for SaaS platform
- Conducted user research and A/B testing
- Worked closely with Python and React engineering teams

[5] resume_megan_green.txt (SUMMARY)
AI research engineer with 4 years of experience in deep learning and computer vision.
```

</details>

**Grade:** 

---

### g18: Summarise Nicole Harris's game development experience.

**Answer:**

> Nicole Harris has four years of experience as a game developer.  
> - From 2020 to 2022 she worked as a Junior Game Developer at IndieGames Co., where she built 2D platformers in Unity, created UI systems and inventory management, and designed levels and gameplay mechanics (resume_nicole_harris.txt).  
> - Since 2022 she has been a Game Developer at PixelForge Studios, developing 3D games with Unity and C#, implementing physics systems and AI behaviors, and optimizing rendering performance for mobile platforms (resume_nicole_harris.txt).

<details><summary>Excerpts the model was given</summary>

```
[1] resume_nicole_harris.txt (HEADER)
NICOLE HARRIS
Game Developer

[2] resume_nicole_harris.txt (SUMMARY)
Game developer with 4 years of experience in game design and development.

[3] resume_nicole_harris.txt (EXPERIENCE)
Game Developer | PixelForge Studios | 2022 - Present
- Developed 3D games using Unity and C#
- Implemented physics systems and AI behaviors
- Optimized rendering performance for mobile platforms

Junior Game Developer | IndieGames Co. | 2020 - 2022
- Built 2D platformers using Unity
- Created game UI systems and inventory management
- Designed levels and gameplay mechanics

[4] resume_alex_kumar.txt (SUMMARY)
Full stack developer with 3 years of experience building web applications
using JavaScript, React, and Node.js. Strong background in agile development.

[5] resume_chris_lee.txt (SUMMARY)
Mobile developer with 4 years of experience building iOS and Android apps.
Proficient in Swift, Kotlin, and cross-platform development with Flutter.
```

</details>

**Grade:** 

---

### g19: Which candidates have more than 5 years of experience?

**Answer:**

> Candidates with more than 5 years of experience:
> 
> - Andrew Young – 8 years of experience (resume_andrew_young.txt)  
> - John Doe – 5+ years of experience (resume_john_doe.txt)

<details><summary>Excerpts the model was given</summary>

```
[1] resume_daniel_johnson.txt (SUMMARY)
QA engineer with 5 years of experience in test automation and quality assurance.

[2] resume_jason_wright.txt (SUMMARY)
Embedded engineer with 5 years of experience in firmware and IoT systems.

[3] resume_andrew_young.txt (SUMMARY)
Solutions architect with 8 years of experience designing enterprise-scale systems.

[4] resume_derek_campbell.txt (SUMMARY)
Salesforce developer with 5 years of experience building custom CRM solutions
and integrations for enterprise clients.

[5] resume_john_doe.txt (SUMMARY)
Experienced software engineer with 5+ years of expertise in Python, JavaScript,
and cloud technologies. Passionate about building scalable applications.
```

</details>

**Grade:** 

---

### g20: Who is the strongest candidate for a senior DevOps role, and why?

**Answer:**

> The only candidate in the excerpts who has a senior‑level DevOps background is **Maria Garcia**.
> 
> **Why Maria stands out**
> 
> | Evidence | Why it matters for a senior DevOps role |
> |----------|----------------------------------------|
> | **Summary** – “DevOps engineer with 6 years of experience in cloud infrastructure, CI/CD pipelines, and container orchestration. AWS and Azure certified.” (resume_maria_garcia.txt) | Demonstrates breadth of DevOps skills and industry‑standard certifications. |
> | **Senior DevOps Engineer at CloudScale Inc. (2021‑Present)** – “Managed Kubernetes clusters serving 10M+ requests daily; automated infrastructure using Terraform and Ansible; reduced deployment time by 70% with optimized CI/CD pipelines.” (resume_maria_garcia.txt) | Shows hands‑on leadership of large‑scale Kubernetes environments, IaC expertise, and tangible performance improvements. |
> | **DevOps Engineer at InfraTech (2018‑2021)** – “Set up monitoring with Prometheus, Grafana, and ELK stack; migrated on‑premise applications to AWS; implemented Docker containerization for 20+ services.” (resume_maria_garcia.txt) | Highlights end‑to‑end DevOps responsibilities: monitoring, cloud migration, and containerization. |
> 
> **Comparison with other candidates**
> 
> - Derek Campbell is a Senior Salesforce Developer, not a DevOps engineer.  
> - Andrew Young is a Principal/Senior Solutions Architect, focused on cloud architecture rather than day‑to‑day DevOps operations.
> 
> Given the explicit senior DevOps experience, proven impact on deployment efficiency, and comprehensive skill set, Maria Garcia is the strongest candidate for a senior DevOps role.

<details><summary>Excerpts the model was given</summary>

```
[1] resume_derek_campbell.txt (EXPERIENCE)
Senior Salesforce Developer | CRMPro Solutions | 2021 - Present
- Developed custom Salesforce applications using Apex and Lightning Web Components
- Built integrations with external systems using REST APIs and MuleSoft
- Managed data migrations for 500K+ records using Python and Data Loader

Salesforce Developer | CloudCRM Inc. | 2019 - 2021
- Customized Salesforce workflows and process builders
- Created Visualforce pages and Lightning components
- Wrote unit tests achieving 90%+ code coverage

[2] resume_maria_garcia.txt (SUMMARY)
DevOps engineer with 6 years of experience in cloud infrastructure,
CI/CD pipelines, and container orchestration. AWS and Azure certified.

[3] resume_andrew_young.txt (EXPERIENCE)
Principal Solutions Architect | EnterpriseCloud | 2020 - Present
- Designed cloud-native architectures for Fortune 500 clients
- Led migration of monolithic systems to microservices on AWS
- Reduced infrastructure costs by 45% through optimization

Senior Solutions Architect | TechBridge | 2016 - 2020
- Architected event-driven systems using Kafka and AWS Lambda
- Designed multi-tenant SaaS platforms
- Mentored junior architects and engineers

[4] resume_maria_garcia.txt (HEADER)
MARIA GARCIA
DevOps Engineer

[5] resume_maria_garcia.txt (EXPERIENCE)
Senior DevOps Engineer | CloudScale Inc. | 2021 - Present
- Managed Kubernetes clusters serving 10M+ requests daily
- Automated infrastructure using Terraform and Ansible
- Reduced deployment time by 70% with optimized CI/CD pipelines

DevOps Engineer | InfraTech | 2018 - 2021
- Set up monitoring with Prometheus, Grafana, and ELK stack
- Migrated on-premise applications to AWS
- Implemented Docker containerization for 20+ services
```

</details>

**Grade:** 

---

### g21: Did Rachel Moore ever work at Google?

**Answer:**

> The résumés provided do not say.

<details><summary>Excerpts the model was given</summary>

```
[1] resume_rachel_moore.txt (HEADER)
RACHEL MOORE
NLP Engineer

[2] resume_michelle_king.txt (EDUCATION)
B.A. English | University of Oregon | 2020
Google Technical Writing Certificate

[3] resume_chris_lee.txt (EXPERIENCE)
Senior Mobile Developer | AppWorks Studio | 2022 - Present
- Developed iOS apps using Swift and SwiftUI
- Built cross-platform apps with Flutter and Dart
- Integrated Firebase for authentication and real-time databases

Mobile Developer | MobileFirst Inc. | 2020 - 2022
- Created Android apps using Kotlin and Jetpack Compose
- Implemented push notifications and in-app purchases
- Published 5 apps on App Store and Google Play

[4] resume_olivia_baker.txt (HEADER)
OLIVIA BAKER
Business Intelligence Analyst

[5] resume_james_adams.txt (EXPERIENCE)
React Native Developer | MobileApps Co. | 2022 - Present
- Built cross-platform mobile apps using React Native and Expo
- Integrated native modules for iOS and Android
- Implemented state management with Redux and MobX

Junior Mobile Developer | AppStudio | 2021 - 2022
- Developed UI components using React Native
- Integrated REST APIs and Firebase
- Published 4 apps on App Store and Google Play
```

</details>

**Grade:** 

---

### g22: Which open-source projects has William Carter contributed to?

**Answer:**

> The résumé excerpts for William Carter state that he “contributed to open‑source Rust libraries” (resume_william_carter.txt). No specific project names are listed.

<details><summary>Excerpts the model was given</summary>

```
[1] resume_william_carter.txt (HEADER)
WILLIAM CARTER
Rust Systems Programmer

[2] resume_william_carter.txt (EXPERIENCE)
Systems Engineer | PerformanceTech | 2022 - Present
- Developed high-throughput networking libraries in Rust
- Built memory-safe concurrent systems
- Optimized performance-critical code paths

Junior Systems Developer | LowLevel Inc. | 2021 - 2022
- Wrote systems software in C++ and Rust
- Implemented custom allocators and data structures
- Contributed to open-source Rust libraries

[3] resume_andrew_young.txt (HEADER)
ANDREW YOUNG
Solutions Architect

[4] resume_james_adams.txt (HEADER)
JAMES ADAMS
React Native Developer

[5] resume_christopher_lee2.txt (HEADER)
CHRISTOPHER LEE
Go Backend Developer
```

</details>

**Grade:** 

---

### g23: Which candidates have led or managed engineering teams?

**Answer:**

> The résumé excerpts indicate that **Jennifer White** has led or managed engineering teams.  
> - In her role as Senior Product Manager at ProductHub, she “managed team of 8 engineers and 2 designers”【1】.  
> 
> No other candidate’s excerpts mention leading or managing engineering teams.

<details><summary>Excerpts the model was given</summary>

```
[1] resume_jennifer_white.txt (EXPERIENCE)
Senior Product Manager | ProductHub | 2021 - Present
- Led roadmap for ML-powered recommendation engine
- Managed team of 8 engineers and 2 designers
- Increased user retention by 25% through feature optimization

Product Manager | TechStartup | 2018 - 2021
- Defined product requirements for SaaS platform
- Conducted user research and A/B testing
- Worked closely with Python and React engineering teams

[2] resume_jennifer_white.txt (SUMMARY)
Technical product manager with 6 years of experience leading cross-functional teams
and delivering data-driven products. Background in software engineering.

[3] resume_jason_wright.txt (EDUCATION)
M.S. Electrical Engineering | University of Michigan | 2019
B.S. Electrical Engineering | Michigan State University | 2017

[4] resume_andrew_young.txt (EDUCATION)
M.S. Software Engineering | University of North Carolina | 2016
B.S. Computer Science | NC State University | 2014
AWS Solutions Architect Professional

[5] resume_john_doe.txt (HEADER)
JOHN DOE
Software Engineer
```

</details>

**Grade:** 

---

### g24: What is Victoria Ross's experience with deploying models to edge devices?

**Answer:**

> Victoria Ross has deployed models on edge devices using TensorRT and ONNX. (resume_victoria_ross.txt)

<details><summary>Excerpts the model was given</summary>

```
[1] resume_victoria_ross.txt (HEADER)
VICTORIA ROSS
Computer Vision Engineer

[2] resume_andrew_young.txt (SUMMARY)
Solutions architect with 8 years of experience designing enterprise-scale systems.

[3] resume_marcus_cooper.txt (EXPERIENCE)
MLOps Engineer | MLScale Inc. | 2022 - Present
- Built ML pipelines using Kubeflow and MLflow on Kubernetes
- Automated model training, evaluation, and deployment workflows
- Implemented feature stores using Feast and Redis

Junior MLOps Engineer | DataOps Co. | 2021 - 2022
- Containerized ML models using Docker
- Set up model monitoring with Prometheus and Grafana
- Managed GPU clusters on AWS SageMaker and GCP Vertex AI

[4] resume_victoria_ross.txt (EXPERIENCE)
Computer Vision Engineer | AutoVision Labs | 2022 - Present
- Developed real-time object detection models using YOLOv8 and PyTorch
- Built 3D point cloud processing pipelines using Open3D
- Deployed models on edge devices using TensorRT and ONNX

Junior CV Engineer | ImageTech | 2021 - 2022
- Trained image classification models using TensorFlow and Keras
- Created data augmentation pipelines for training datasets
- Implemented image preprocessing with OpenCV and Pillow

[5] resume_brian_jackson.txt (SUMMARY)
iOS developer with 4 years of experience building native Apple applications.
```

</details>

**Grade:** 

