import csv

def predict_personality(row):
    """Predict personality based on features"""
    # Extract features
    time_alone = float(row.get('Time_spent_Alone', 0) or 0)
    stage_fear = row.get('Stage_fear', 'No')
    social_events = float(row.get('Social_event_attendance', 0) or 0)
    going_outside = float(row.get('Going_outside', 0) or 0)
    drained = row.get('Drained_after_socializing', 'No')
    friends_size = float(row.get('Friends_circle_size', 0) or 0)
    post_freq = float(row.get('Post_frequency', 0) or 0)
    
    # Calculate engineered features
    social_activity_score = social_events + going_outside + post_freq
    social_energy = time_alone * 0.5 + (2 if stage_fear == 'Yes' else 0) + (2 if drained == 'Yes' else 0)
    introversion_score = time_alone * 0.3 + (1.5 if stage_fear == 'Yes' else 0) + (1.5 if drained == 'Yes' else 0) - social_events * 0.2
    extroversion_score = social_events * 0.3 + going_outside * 0.3 + post_freq * 0.2 + friends_size * 0.2 - time_alone * 0.1
    
    # Decision logic
    if social_activity_score > 15 and extroversion_score > 2:
        return "Extrovert"
    elif introversion_score > 3 or social_energy > 5:
        return "Introvert"
    elif friends_size > 10 and social_events > 5:
        return "Extrovert"
    elif time_alone > 6 or stage_fear == 'Yes':
        return "Introvert"
    elif post_freq > 7 and social_events > 6:
        return "Extrovert"
    elif drained == 'Yes' and time_alone > 3:
        return "Introvert"
    else:
        return "Extrovert" if social_activity_score > 10 else "Introvert"

def generate_complete_submission():
    """Generate complete submission file"""
    
    predictions = []
    
    # Read test.csv and generate predictions
    with open('test.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            test_id = int(row['id'])
            prediction = predict_personality(row)
            predictions.append([test_id, prediction])
    
    # Write complete submission file
    with open('optimized_submission.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'Personality'])
        writer.writerows(predictions)
    
    print(f"Created complete optimized_submission.csv with {len(predictions)} predictions")
    
    # Print statistics
    extrovert_count = sum(1 for _, p in predictions if p == 'Extrovert')
    introvert_count = sum(1 for _, p in predictions if p == 'Introvert')
    
    print(f"\nPrediction Statistics:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Extrovert: {extrovert_count} ({extrovert_count/len(predictions)*100:.1f}%)")
    print(f"Introvert: {introvert_count} ({introvert_count/len(predictions)*100:.1f}%)")
    
    # Show sample predictions
    print(f"\nFirst 10 predictions:")
    for i, (id_val, personality) in enumerate(predictions[:10]):
        print(f"{id_val},{personality}")
    
    print(f"\nLast 10 predictions:")
    for i, (id_val, personality) in enumerate(predictions[-10:]):
        print(f"{id_val},{personality}")

if __name__ == "__main__":
    print("Generating complete optimized_submission.csv...")
    generate_complete_submission()
    print("Done!") 