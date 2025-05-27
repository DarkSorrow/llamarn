import React, { useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  Button,
  ScrollView,
  SafeAreaView,
  Platform,
} from 'react-native';
import { multiply } from '@novastera-oss/llamarn';
import ConsolidatedTestScreen from './ConsolidatedTestScreen';
import ModelChatTestScreen from './ModelChatTestScreen';

// Placeholder for the actual test components we will create later
// const ConsolidatedTestScreen = () => ( <--- THIS LINE AND ITS BODY NEED TO BE DELETED
// <View style={styles.pageContainer}>
// <Text style={styles.pageTitle}>Consolidated Test Screen</Text>
// <Text>Functionality for consolidated tests will be implemented here.</Text>
// {/* We will later integrate model loading, info display, and various tests here */}
// </View>
// );

type Screen = 'menu' | 'consolidatedTest' | 'modelChatTest';

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<Screen>('menu');

  const renderContent = () => {
    switch (currentScreen) {
      case 'consolidatedTest':
        return <ConsolidatedTestScreen />;
      case 'modelChatTest':
        return <ModelChatTestScreen />;
      case 'menu':
      default:
        return (
          <View style={styles.menuContainer}>
            <Text style={styles.title}>Llama.rn Test Menu</Text>
            <View style={styles.buttonWrapper}>
              <Button
                title="Go to Consolidated Test"
                onPress={() => setCurrentScreen('consolidatedTest')}
              />
            </View>
            <View style={styles.buttonWrapper}>
              <Button
                title="Go to Model Chat Test"
                onPress={() => setCurrentScreen('modelChatTest')}
              />
            </View>
            <View style={styles.infoBox}>
              <Text style={styles.infoText}>
                Base Library Sanity Check:
              </Text>
              <Text style={styles.infoText_mono_small}>
                multiply(5, 4) = {multiply(5, 4)}
              </Text>
            </View>
          </View>
        );
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollViewContainer}>
        {currentScreen !== 'menu' && (
          <View style={styles.backButtonContainer}>
            <Button title="Back to Menu" onPress={() => setCurrentScreen('menu')} />
          </View>
        )}
        {renderContent()}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f0f0f0',
  },
  scrollViewContainer: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  menuContainer: {
    width: '100%',
    alignItems: 'center',
  },
  pageContainer: {
    flex: 1,
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 30,
    textAlign: 'center',
  },
  pageTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  buttonWrapper: {
    width: '80%',
    marginVertical: 10,
  },
  infoBox: {
    marginTop: 40,
    padding: 15,
    backgroundColor: '#e7e7e7',
    borderRadius: 8,
    width: '90%',
    alignItems: 'center',
  },
  infoText: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 5,
  },
  infoText_mono_small: {
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    fontSize: 14,
    textAlign: 'center',
  },
  backButtonContainer: {
    position: 'absolute',
    top: 20, // Adjust as needed for status bar height
    left: 20,
    zIndex: 10, // Ensure it's above other content
  },
});
