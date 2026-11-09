PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS paper(
    paper_id INTEGER PRIMARY KEY,
    authors TEXT NOT NULL,
    year INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model(
    model_id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL,
    equation TEXT NOT NULL,
    equation_family TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (paper_id) REFERENCES paper(paper_id)
);

CREATE TABLE IF NOT EXISTS relationship(
    relation_id INTEGER PRIMARY KEY,
    parent_model_id INTEGER NOT NULL,
    child_model_id INTEGER NOT NULL,
    relation TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (parent_model_id) REFERENCES model(model_id),
    FOREIGN KEY (child_model_id) REFERENCES model(model_id)
);

-- INSERT INTO paper(paper_id, authors, year)
-- VALUES
--     (1, 'Iyad Salam Saidi', '1992'),
--     (2, 'Steven L. Jacques', '1998'),
--     (3, 'Rong Zhang; Wim Verkruysse; Bernard Choi; John A. Viator; Byungjo Jung; Lars O. Svaasand; Guillermo Aguilar; J. Stuart Nelson', '2005');



-- INSERT INTO model(model_id, paper_id, equation, equation_family)
-- VALUES
--     (
--         1,
--         1,
--         'mu_a_baseline(lambda) = 7.84e8 * lambda^(-3.255) [cm^-1]',
--         'power_law'
--     ),
--     (
--         2,
--         2,
--         'mu_a_baseline(lambda) = 7.84e8 * lambda^(-3.255) [cm^-1]',
--         'power_law'
--     ),
--     (
--         3,
--         3,
--         'mu_a_T(lambda) = 7.84e7 * lambda^(-3.255) [mm^-1]',
--         'power_law'
--     );

-- INSERT INTO relationship(relation_id, parent_model_id, child_model_id, relation)
-- VALUES
--     (
--         1,
--         1,
--         2,
--         'SAIDI_POWER_LAW_REPORTED'
--     ),
--     (
--         2,
--         1,
--         3,
--         'POWER_LAW_ADOPTED'
--     );