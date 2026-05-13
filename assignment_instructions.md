- Klassen bygger en helt automatiserad SW-developer organisation, med en agent per student, och en central hub för kommunikation

- Förberedande lektion kring IT-säkerhet kring agenter, hur man undviker att agenten gör slut på alla cloud credits vid API-anrop eller orsakar stora kostnader, 
eller osäkra kommandoexekveringar

- Agenterna baseras helt och hållet på Python-kod och API-anrop till OpenAI Platform API, Anthropic API, OpenRouter, lokala modeller, eller liknande.

- Ingen OpenCode, KiloCode, Claude-Code, Cursor, Anti-gravity, Codex, etc får vara del av agentens funktion. Agenterna ska helt och hållet vara Python-baserade med egen kod. Ni får använda ramverk som LangGraph, LangChain, LlamaIndex, etc endast i de delar där det uttryckligen tillåts i Assignment-2, d.v.s. håll noga rätt på vad reglerna är för respektive del-uppgift. Anledningen är att man behöver kunna grunderna innan man gömmer undan allt bakom ett ramverk - men ramverken är också bra att kunna därefter.


Del 1: Bygg en ReAct agent, med bash commands via hemmagjord function-calling (inga ramverk, ingen inbyggd function-calling).
I del 1 är det krav att ni använder rå text-output och gör er egen stränghantering. Syftet är att ni ska se hur lätt det är att bygga en sådan harness från scratch.

Del 2: I del 2 är tanken att ni ska byta till mainstream structured output, och bygga en starkare variant. Kravet är endast att det är er egen kod som står för agent-loopen, context-hanteringen, tool-calling, etc. Men parsing av output kan göras på valfritt sätt. Kravet i del 2 är att ert program kan:
bash-anrop med säkerhetsspärr mot destruktiva eller skadliga exekveringar (fortsatt rekommenderat att köra i container ändå!)
editering av enskilda avsnitt av filer
multipla tool-calling rounds innan yield till användaren. Modellen avgör själv yield eller tool-call.
persistent lagring av session history inom sessionen (multi-session ej krav).
System-prompt från config-fil. Sys-prompten ska styra AIn att endast jobba på säkert sätt med SWE (software engineering), och avböja andra ämnen.
Tool-calling ska ha begränsning m.a.p. storlek på output från verktyget, och agenten ska känna till begränsningen.

Del 3: I del 3 ska er agent överföra kod mellan sig själv och andra agenter, samarbeta konstruktivt och meningsfullt i ett gemensamt mjukvaru-projekt jag kommer att meddela på lektion.
Sys-prompten ska instruera agenten att inte läcka känslig information till andra agenter.
Agenten ska agera ansvarsfullt gentemot andra agenter och vara en "team-player" och respektera överenskomna samarbetsformer - men samarbetsformerna bestäms av agenterna, och kan bli olika vid olika tillfällen.
Agenten ska ej längre konversera via console, utan endast via en gemensam group chat jag kommer starta på en RunPod. Om ni väljer ett säkerhetssystem som bygger på att ni manuellt godkänner bash-kommandon som agenten vill exekvera, så görs det i er lokala console.
Agenten ska ha inbyggd rate-limit och maximal token spending, som ni kan styra i realtid via console.
Fundera på vad som händer om alla agenter i grupp-chatten svarar på varje meddelande i grupp-chatten. Designa något smart utifrån ert eget svar på den frågan.