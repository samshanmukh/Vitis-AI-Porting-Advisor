<br><br><br> 
<h2 align="center">SSLChange Method Global Inference diagram architecture</h2> 
 
mermaid 
flowchart TB 
 
    A1["Image T0"] --> B1["Clipped Backbone"] 
    A2["Image T1"] --> B2["Clipped Backbone"] 
 
    B1 --> C1["Up module (Alignment)"] 
    B2 --> C2["Up module (Alignment)"] 
 
    C1 --> D["Concatenation with input images"] 
    C2 --> D 
 
    D --> E["Change detection classifier<br/>(USSFCNet model)"] 
    E --> F["Change map"]