- Klassen bygger en helt automatiserad SW-developer organisation, med en agent per student, och en central hub för kommunikation

- Förberedande lektion kring IT-säkerhet kring agenter, hur man undviker att agenten gör slut på alla cloud credits vid API-anrop eller orsakar stora kostnader, 
eller osäkra kommandoexekveringar

- Agenterna baseras helt och hållet på Python-kod och API-anrop till OpenAI Platform API, Anthropic API, OpenRouter, lokala modeller, eller liknande.

- Ingen OpenCode, KiloCode, Claude-Code, Cursor, Anti-gravity, Codex, etc får vara del av agentens funktion. Agenterna ska helt och hållet vara Python-baserade med egen kod. Ni får använda ramverk som LangGraph, LangChain, LlamaIndex, etc endast i de delar där det uttryckligen tillåts i Assignment-2, d.v.s. håll noga rätt på vad reglerna är för respektive del-uppgift. Anledningen är att man behöver kunna grunderna innan man gömmer undan allt bakom ett ramverk - men ramverken är också bra att kunna därefter.


Del 1: Bygg en ReAct agent, med bash commands via hemmagjord function-calling (inga ramverk, ingen inbyggd function-calling) 