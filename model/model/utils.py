def print_label_mappings(label_encoder_level, label_encoder_category):
    risk_level_mapping = {i: label for i, label in enumerate(label_encoder_level.classes_)}
    risk_category_mapping = {i: label for i, label in enumerate(label_encoder_category.classes_)}
    
    print("Risk Level Encodings:")
    for encoded, original in risk_level_mapping.items():
        print(f"{original}: {encoded}")
        
    print("\nRisk Category Encodings:")
    for encoded, original in risk_category_mapping.items():
        print(f"{original}: {encoded}")