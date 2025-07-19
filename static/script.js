// Personality Prediction Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('personalityForm');
    const resultsSection = document.getElementById('resultsSection');
    
    form.addEventListener('submit', handleFormSubmit);
});

function handleFormSubmit(event) {
    event.preventDefault();
    
    // Show loading state
    const submitBtn = event.target.querySelector('.predict-btn');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<span class="loading"></span> Analyzing...';
    submitBtn.disabled = true;
    
    // Get form data
    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData);
    
    // Simulate API call delay
    setTimeout(() => {
        const prediction = predictPersonality(data);
        displayResults(prediction);
        
        // Reset button
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }, 2000);
}

function predictPersonality(data) {
    // Convert form data to model features
    const features = {
        'Time_spent_Alone': parseFloat(data.timeAlone),
        'Stage_fear': data.stageFear === 'Yes' ? 1 : 0,
        'Social_event_attendance': parseInt(data.socialEvents),
        'Going_outside': parseInt(data.goingOutside),
        'Drained_after_socializing': data.drainedAfterSocializing === 'Yes' ? 1 : 0,
        'Friends_circle_size': parseInt(data.friendsCircle),
        'Post_frequency': parseInt(data.postFrequency)
    };
    
    // Debug logging
    console.log('Input features:', features);
    
    // Calculate derived features (matching the model's feature engineering)
    const socialActivityScore = features['Social_event_attendance'] + features['Going_outside'] + features['Post_frequency'];
    const socialEnergy = features['Time_spent_Alone'] * 0.5 + features['Stage_fear'] * 2 + features['Drained_after_socializing'] * 2;
    const introversionScore = features['Time_spent_Alone'] * 0.3 + features['Stage_fear'] * 1.5 + features['Drained_after_socializing'] * 1.5 - features['Social_event_attendance'] * 0.2;
    const extroversionScore = features['Social_event_attendance'] * 0.3 + features['Going_outside'] * 0.3 + features['Post_frequency'] * 0.2 + features['Friends_circle_size'] * 0.2 - features['Time_spent_Alone'] * 0.1;
    
    console.log('Scores:', { introversionScore, extroversionScore });
    
    // Simple prediction logic based on the model's feature importance
    const introversionIndicators = [
        features['Time_spent_Alone'] > 8,
        features['Stage_fear'] === 1,
        features['Drained_after_socializing'] === 1,
        features['Social_event_attendance'] < 2,
        features['Going_outside'] <= 2,
        features['Friends_circle_size'] <= 2,
        features['Post_frequency'] < 3
    ];
    
    const extroversionIndicators = [
        features['Time_spent_Alone'] < 4,
        features['Stage_fear'] === 0,
        features['Drained_after_socializing'] === 0,
        features['Social_event_attendance'] > 4,
        features['Going_outside'] >= 3,
        features['Friends_circle_size'] >= 3,
        features['Post_frequency'] > 5
    ];
    
    console.log('Introversion indicators:', introversionIndicators);
    console.log('Extroversion indicators:', extroversionIndicators);
    
    const introversionScore_count = introversionIndicators.filter(Boolean).length;
    const extroversionScore_count = extroversionIndicators.filter(Boolean).length;
    
    console.log('Indicator counts:', { introversionScore_count, extroversionScore_count });
    
    // Calculate confidence based on feature alignment
    const totalIndicators = introversionIndicators.length;
    const introversionConfidence = (introversionScore_count / totalIndicators) * 100;
    const extroversionConfidence = (extroversionScore_count / totalIndicators) * 100;
    
    console.log('Confidence levels:', { introversionConfidence, extroversionConfidence });
    
    // Determine personality type
    let personalityType, confidence, traits, explanation;
    
    if (introversionScore > extroversionScore || introversionScore_count > extroversionScore_count) {
        personalityType = 'Introvert';
        confidence = Math.max(introversionConfidence, 60);
        traits = getIntrovertTraits(features);
        explanation = getIntrovertExplanation();
    } else {
        personalityType = 'Extrovert';
        confidence = Math.max(extroversionConfidence, 60);
        traits = getExtrovertTraits(features);
        explanation = getExtrovertExplanation();
    }
    
    console.log('Final prediction:', { personalityType, confidence });
    
    return {
        personalityType,
        confidence: Math.min(confidence, 95), // Cap at 95% for realism
        traits,
        explanation,
        features
    };
}

function getIntrovertTraits(features) {
    const traits = [];
    
    if (features['Time_spent_Alone'] > 8) {
        traits.push({ trait: 'Solitude Preference', description: 'Enjoys significant alone time' });
    }
    if (features['Stage_fear'] === 1) {
        traits.push({ trait: 'Public Speaking Anxiety', description: 'Prefers smaller, intimate settings' });
    }
    if (features['Drained_after_socializing'] === 1) {
        traits.push({ trait: 'Social Energy Conservation', description: 'Needs recovery time after social interactions' });
    }
    if (features['Social_event_attendance'] < 2) {
        traits.push({ trait: 'Selective Socializing', description: 'Chooses quality over quantity in social interactions' });
    }
    if (features['Friends_circle_size'] <= 2) {
        traits.push({ trait: 'Deep Relationships', description: 'Values close, meaningful friendships' });
    }
    
    return traits.length > 0 ? traits : [
        { trait: 'Thoughtful', description: 'Prefers to think before speaking' },
        { trait: 'Observant', description: 'Notices details others might miss' }
    ];
}

function getExtrovertTraits(features) {
    const traits = [];
    
    if (features['Time_spent_Alone'] < 4) {
        traits.push({ trait: 'Social Energy', description: 'Gains energy from social interactions' });
    }
    if (features['Stage_fear'] === 0) {
        traits.push({ trait: 'Confident Communicator', description: 'Comfortable in the spotlight' });
    }
    if (features['Social_event_attendance'] > 4) {
        traits.push({ trait: 'Social Butterfly', description: 'Actively seeks social opportunities' });
    }
    if (features['Friends_circle_size'] >= 3) {
        traits.push({ trait: 'Network Builder', description: 'Maintains a wide social network' });
    }
    if (features['Post_frequency'] > 5) {
        traits.push({ trait: 'Social Media Active', description: 'Enjoys sharing experiences online' });
    }
    
    return traits.length > 0 ? traits : [
        { trait: 'Outgoing', description: 'Naturally engages with others' },
        { trait: 'Enthusiastic', description: 'Brings energy to social situations' }
    ];
}

function getIntrovertExplanation() {
    return "Introverts typically prefer smaller, more intimate social settings and often need time alone to recharge. They tend to think before speaking and may prefer deep, meaningful conversations over small talk. This doesn't mean they're shy - many introverts are quite confident and social, but they prefer quality over quantity in their social interactions.";
}

function getExtrovertExplanation() {
    return "Extroverts are energized by social interactions and often seek out opportunities to connect with others. They tend to think out loud and may prefer to work through ideas by talking them through. Extroverts typically have wide social networks and enjoy being the center of attention in group settings.";
}

function displayResults(prediction) {
    const resultsSection = document.getElementById('resultsSection');
    const personalityType = document.getElementById('personalityType');
    const personalityBadge = document.getElementById('personalityBadge');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    const traitsGrid = document.getElementById('traitsGrid');
    const explanationText = document.getElementById('explanationText');
    
    // Set personality type
    personalityType.textContent = prediction.personalityType;
    personalityBadge.className = `personality-badge ${prediction.personalityType.toLowerCase()}`;
    
    // Set confidence
    confidenceFill.style.width = `${prediction.confidence}%`;
    confidenceText.textContent = `${Math.round(prediction.confidence)}%`;
    
    // Set traits
    traitsGrid.innerHTML = '';
    prediction.traits.forEach(trait => {
        const traitElement = document.createElement('div');
        traitElement.className = 'trait-item';
        traitElement.innerHTML = `
            <strong>${trait.trait}</strong><br>
            <span>${trait.description}</span>
        `;
        traitsGrid.appendChild(traitElement);
    });
    
    // Set explanation
    explanationText.textContent = prediction.explanation;
    
    // Show results with animation
    resultsSection.style.display = 'block';
    resultsSection.style.opacity = '0';
    resultsSection.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        resultsSection.style.opacity = '1';
        resultsSection.style.transform = 'translateY(0)';
    }, 100);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function resetForm() {
    const form = document.getElementById('personalityForm');
    const resultsSection = document.getElementById('resultsSection');
    
    // Hide results
    resultsSection.style.opacity = '0';
    resultsSection.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        resultsSection.style.display = 'none';
        form.reset();
        
        // Scroll to form
        form.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 300);
}

// Add some interactive features
document.addEventListener('DOMContentLoaded', function() {
    // Add input validation
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            const value = parseFloat(this.value);
            const min = parseFloat(this.min);
            const max = parseFloat(this.max);
            
            if (value < min) this.value = min;
            if (value > max) this.value = max;
        });
    });
    
    // Add form field highlighting
    const formFields = document.querySelectorAll('.form-group input, .form-group select');
    formFields.forEach(field => {
        field.addEventListener('focus', function() {
            this.parentElement.style.transform = 'scale(1.02)';
        });
        
        field.addEventListener('blur', function() {
            this.parentElement.style.transform = 'scale(1)';
        });
    });
}); 