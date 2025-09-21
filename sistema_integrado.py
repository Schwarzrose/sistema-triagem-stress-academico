'''Sistema Integrado de Triagem - Stress Acadêmico
Executa o classificador (Random Forest) e o regressor (SVR) para validação cruzada a fim de auxiliar na identificação e priorização de casos 
que necessitam de suporte psicológico. A proposta é um sistema híbrido de verificação em duas etapas para auxiliar instituições de ensino.    
'''

import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

print("Carregando modelos...")
classificador = joblib.load('modelo_classificador_rf.pkl')
regressor = joblib.load('modelo_regressor_svr.pkl')
scaler = joblib.load('scaler_regressao.pkl')
encoding_info = joblib.load('pandas_encoding_estrutura.pkl')
print("Modelos e estrutura de encoding carregados com sucesso!")

def aplicar_encoding_pandas(academic_stage, peer_pressure, academic_pressure_home, 
                                 study_environment, coping_strategy, bad_habits, 
                                 academic_competition_level):

#Réplica do processo de encoding usando o pandas feito nos projetos anteriores.

    df_temp = pd.DataFrame({
        'peer_pressure': [peer_pressure],
        'academic_pressure_home': [academic_pressure_home],
        'academic_competition_level': [academic_competition_level],
        'academic_stage': [academic_stage],
        'study_environment': [study_environment],
        'coping_strategy': [coping_strategy],
        'bad_habits': [bad_habits]
    })
    
    for c in encoding_info['colunas_categoricas']:
        df_temp[c] = df_temp[c].astype('string')
    
    
    df_encoded = pd.get_dummies(df_temp, columns=encoding_info['colunas_categoricas'], drop_first=True)
    
    colunas_esperadas = encoding_info['colunas_finais_x']
    
    for coluna in colunas_esperadas:
        if coluna not in df_encoded.columns:
            df_encoded[coluna] = 0

    df_encoded = df_encoded[colunas_esperadas]
    
    return df_encoded.values[0] 

def sistema_triagem(academic_stage, peer_pressure, academic_pressure_home, 
                   study_environment, coping_strategy, bad_habits, 
                   academic_competition_level):
    '''
Função principal que executa o fluxo de 2 etapas:
Primeiro o classificador categoriza o stress do estudante;
Em seguida o regressor determina um nível (entre 1 a 5) de stress;
Enfim a validação cruzada aponta uma sugestão após a tomada de decisão.
    '''
    
    print("SISTEMA DE TRIAGEM ACADÊMICA:")
    print("=" * 50)
    
    dados_encoded = aplicar_encoding_pandas(
        academic_stage, peer_pressure, academic_pressure_home,
        study_environment, coping_strategy, bad_habits, 
        academic_competition_level
    )
    
    
    #Classificação (Random Forest)
    classe_pred = classificador.predict([dados_encoded])[0]
    classe_proba = classificador.predict_proba([dados_encoded])[0]
    
    print(f"ETAPA 1 - Classificador (Random Forest):")
    print(f"Predição: {classe_pred}")
    print(f"Confiança: {max(classe_proba):.3f}")
    
    #Regressão (SVR)
    dados_scaled = scaler.transform([dados_encoded])
    score_pred = regressor.predict(dados_scaled)[0]
    score_pred = np.clip(score_pred, 1.0, 5.0)  #Limitador
    
    print(f"ETAPA 2 - Regressor (SVR):")
    print(f"Score: {score_pred:.2f}")
    
    #Validação cruzada e decisão final
    print(f"\nANÁLISE DE CONSENSO:")
    
    #Estabelece relação entre classe e intervalo numérico
    classe_ranges = {
        'Baixo': (1.0, 2.5),
        'Moderado': (2.5, 3.5), 
        'Alto': (3.5, 5.0)
    }
    
    range_esperado = classe_ranges[classe_pred]
    score_dentro_range = range_esperado[0] <= score_pred <= range_esperado[1]
    
    if score_dentro_range:
        decisao = "CONSENSO - Protocolo padrão."
    else:
        if classe_pred == "Alto" and score_pred < 3.5:
            decisao = "DISCORDÂNCIA - Possível falso positivo (revisar)."
        elif classe_pred == "Baixo" and score_pred > 2.5:
            decisao = "DISCORDÂNCIA - Possível falso negativo (atenção especial)."
        else:
            decisao = "DISCORDÂNCIA MODERADA - Monitoramento recomendado."
    
    print(f"   Classe predita: {classe_pred} (esperado: {range_esperado[0]:.1f}-{range_esperado[1]:.1f})")
    print(f"   Score obtido: {score_pred:.2f}")
    print(f"   Decisão: {decisao}")
    
    return {
        'classificacao': classe_pred,
        'confianca': max(classe_proba),
        'score': score_pred,
        'consenso': score_dentro_range,
        'decisao': decisao
    }

#EXEMPLOS DE USO:
if __name__ == "__main__":
    print("Sistema Integrado de Triagem - Stress Acadêmico")
    print("=" * 60)
    
    #EXEMPLO 1: Caso de consenso (stress alto)
    print("\nEXEMPLO 1: Estudante com stress alto")
    resultado1 = sistema_triagem(
        academic_stage='undergraduate',
        peer_pressure=5,
        academic_pressure_home=5, 
        study_environment='disrupted',
        coping_strategy='Emotional breakdown (crying a lot)',
        bad_habits='Yes',
        academic_competition_level=5
    )
    print("-" * 50)
    
    #EXEMPLO 2:Caso ambíguo 
    print("\nEXEMPLO 2: Caso ambíguo")
    resultado2 = sistema_triagem(
        academic_stage='high school',
        peer_pressure=3,
        academic_pressure_home=2,
        study_environment='Peaceful', 
        coping_strategy='Social support (friends, family)',
        bad_habits='No',
        academic_competition_level=2
    )
    print("-" * 50)
    
    #EXEMPLO 3: Caso de stress baixo (Especialmente difícil para o classificador por conta de suas limitações).
    print("\nEXEMPLO 3: Estudante com stress baixo")
    resultado3 = sistema_triagem(
        academic_stage='post-graduate',
        peer_pressure=1,
        academic_pressure_home=1,
        study_environment='Peaceful',
        coping_strategy='Analyze the situation and handle it with intellect', 
        bad_habits='No',
        academic_competition_level=3
    )
    print("-" * 50)
    
    print("\n" + "=" * 60)
    print("RESUMO DOS CASOS TESTADOS:")
    print(f"Exemplo 1: {resultado1['decisao']} - {resultado1['classificacao']}")
    print(f"Exemplo 2: {resultado2['decisao']} - {resultado2['classificacao']}")
    print(f"Exemplo 3: {resultado3['decisao']} - {resultado3['classificacao']}")
    print("\nSistema integrado funcionando!")
    print("Muito obrigado!")