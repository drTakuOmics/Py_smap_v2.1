# TRiC templates

Both models are unpacked automatically by the scripts that use them.

| File | Model |
|---|---|
| `7NVL_mouse.cif.gz` | Closed TRiC: PDB 7NVL (human, closed state) with the eight subunits mutated to the mouse CCT1–8 sequences (248 substitutions). Backbone unchanged (0.0 Å RMSD over 33,672 backbone atoms); substituted side chains rebuilt and energy-minimised. |
| `open_hybrid_mouse.cif.gz` | Open TRiC: the equatorial, intermediate and apical domains of each subunit of the closed mouse model, superposed separately on the open-state scaffold PDB 6NR8 (human TRiC–prefoldin, 7.8 Å; prefoldin not included), hinges relaxed. Domain-fit RMSD: equatorial 1.6–2.6 Å, apical 3–6.5 Å; 50 peptide gaps remain at domain boundaries. |

Check of the open model against PDB 7NVO (human TRiC, open state, 3.5 Å; apical domains not modelled):
2.8 Å Cα RMSD over the 4,406 Cα atoms modelled in 7NVO (equatorial domains 2.4 Å), against 7.5 Å for the
closed model. The apical domain positions come from 6NR8 and represent one conformation of a mobile
region.
