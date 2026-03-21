import React, { useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  Button,
  ScrollView,
  SafeAreaView,
} from 'react-native';
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

  if (currentScreen === 'consolidatedTest' || currentScreen === 'modelChatTest') {
    return (
      <View style={styles.screenContainer}>
        <SafeAreaView style={styles.safeArea}>
          <View style={styles.header}>
            <Button title="Back to Menu" onPress={() => setCurrentScreen('menu')} />
          </View>
          <View style={styles.content}>
            {currentScreen === 'consolidatedTest' ? (
              <ConsolidatedTestScreen />
            ) : (
              <ModelChatTestScreen />
            )}
          </View>
        </SafeAreaView>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollViewContainer}>
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
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f0f0f0',
    paddingTop: 60,
    paddingBottom: 60,
  },
  screenContainer: {
    flex: 1,
    backgroundColor: '#f0f0f0',
  },
  header: {
    paddingHorizontal: 16,
    paddingTop: 8,
    paddingBottom: 8,
  },
  content: {
    flex: 1,
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
});
