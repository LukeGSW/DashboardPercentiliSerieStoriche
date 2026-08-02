"""
=============================================================================
kq.state — Persistenza delle impostazioni fra le modalità dell'app
=============================================================================
Streamlit ELIMINA lo stato di un widget non appena il widget smette di essere
renderizzato. Cambiando modalità, quindi, ogni filtro e ogni parametro
tornerebbe al valore di default.

Un `setdefault` non basta — anzi peggiora le cose, perche' rimette il default
proprio dopo che il valore scelto e' stato eliminato. Le chiavi che NON
appartengono a un widget non vengono invece mai riciclate: si tiene una copia
speculare li' dentro e si reidratano i widget mentre la loro pagina e' nascosta.

La regola e' volutamente binaria, per non dipendere dai tempi con cui Streamlit
ricicla lo stato (non garantiti, e diversi fra versioni):

    pagina A SCHERMO -> non si tocca nulla, altrimenti si annullerebbe ogni
                        modifica appena fatta dall'utente
    pagina NASCOSTA  -> le chiavi vengono tenute idratate dallo specchio
=============================================================================
"""

from __future__ import annotations

import streamlit as st


def ripristina(defaults: dict, prefisso: str, pagina_attiva: bool) -> None:
    """Da chiamare in app.main() PRIMA di istanziare qualunque widget."""
    if pagina_attiva:
        return
    for chiave, default in defaults.items():
        specchio = prefisso + chiave
        st.session_state[chiave] = (
            st.session_state[specchio] if specchio in st.session_state else default
        )


def salva(defaults: dict, prefisso: str) -> None:
    """Da chiamare dopo aver creato i widget della pagina."""
    for chiave in defaults:
        if chiave in st.session_state:
            st.session_state[prefisso + chiave] = st.session_state[chiave]
