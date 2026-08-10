-- ─────────────────────────────────────────────────────────────
--  Seed: Academic Interest Tags
-- ─────────────────────────────────────────────────────────────

INSERT INTO tags (name, category) VALUES

-- Artificial Intelligence
('Machine Learning',        'AI'),
('Deep Learning',           'AI'),
('Natural Language Processing', 'AI'),
('Computer Vision',         'AI'),
('Reinforcement Learning',  'AI'),
('Large Language Models',   'AI'),
('Explainable AI',          'AI'),
('Federated Learning',      'AI'),

-- Data Science
('Data Mining',             'Data Science'),
('Statistical Modeling',    'Data Science'),
('Big Data',                'Data Science'),
('Data Visualization',      'Data Science'),
('Bayesian Methods',        'Data Science'),
('Time Series Analysis',    'Data Science'),

-- Computer Science
('Algorithms',              'Computer Science'),
('Distributed Systems',     'Computer Science'),
('Cybersecurity',           'Computer Science'),
('Blockchain',              'Computer Science'),
('Cloud Computing',         'Computer Science'),
('Quantum Computing',       'Computer Science'),
('Human-Computer Interaction', 'Computer Science'),
('Software Engineering',    'Computer Science'),

-- Biology
('Genomics',                'Biology'),
('CRISPR',                  'Biology'),
('Neuroscience',            'Biology'),
('Bioinformatics',          'Biology'),
('Synthetic Biology',       'Biology'),
('Proteomics',              'Biology'),
('Evolutionary Biology',    'Biology'),

-- Medicine & Health
('Clinical Trials',         'Medicine'),
('Epidemiology',            'Medicine'),
('Medical Imaging',         'Medicine'),
('Drug Discovery',          'Medicine'),
('Mental Health',           'Medicine'),
('Precision Medicine',      'Medicine'),
('Public Health',           'Medicine'),

-- Physics
('Quantum Mechanics',       'Physics'),
('Astrophysics',            'Physics'),
('Particle Physics',        'Physics'),
('Condensed Matter',        'Physics'),
('Thermodynamics',          'Physics'),

-- Mathematics
('Graph Theory',            'Mathematics'),
('Number Theory',           'Mathematics'),
('Topology',                'Mathematics'),
('Optimization',            'Mathematics'),
('Cryptography',            'Mathematics'),
('Differential Equations',  'Mathematics'),

-- Social Sciences
('Behavioral Economics',    'Social Sciences'),
('Cognitive Psychology',    'Social Sciences'),
('Sociology',               'Social Sciences'),
('Political Science',       'Social Sciences'),
('Linguistics',             'Social Sciences'),

-- Environment
('Climate Change',          'Environment'),
('Renewable Energy',        'Environment'),
('Ecology',                 'Environment'),
('Environmental Policy',    'Environment'),

-- Engineering
('Robotics',                'Engineering'),
('Signal Processing',       'Engineering'),
('Materials Science',       'Engineering'),
('Nanotechnology',          'Engineering'),
('Biomedical Engineering',  'Engineering')

ON CONFLICT (name) DO NOTHING;