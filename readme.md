# Evoluzione di Antonia (0-18 anni) - Video Morphing & Selezione IA

Questo progetto permette di partire da un archivio di migliaia di foto sparse sul computer, filtrare automaticamente i ritratti di **Antonia** tramite Face Recognition (InsightFace), scartare foto mosse o non frontali, e assemblare un video in **morphing geometrico fluido** (basato su triangolazione di Delaunay e allineamento pupillare) che mostra la sua evoluzione nel tempo.

---

## 📂 Struttura delle Cartelle del Progetto

Crea una cartella di lavoro sul tuo Mac (es. `EvoluzioneAntonia`) e posiziona al suo interno i file di script e le sottocartelle con questa struttura:

```text
EvoluzioneAntonia/
├── pool_2000_foto/          <-- Inserisci qui tutte le tue foto grezze (supporta sottocartelle)
├── ref_antonia/             <-- Inserisci 3-4 foto nitide di Antonia a età diverse (es. 2, 8, 14, 18 anni)
├── foto_antonia/            <-- Cartella di output dello Script 1 (foto selezionate e ordinate)
├── aligned_debug/           <-- Cartella di supporto per il debug e controllo visivo
├── select_best_faces_id.py  <-- Script 1: Selezione biometrica e filtri di qualità
├── morph_evolution.py       <-- Script 2: Allineamento occhi, triangolazione e video MP4
└── README.md                <-- Questo file di documentazione