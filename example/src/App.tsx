import { Text, View, StyleSheet } from 'react-native';
import { multiply } from '@novastera-oss/llamarn';

const result = multiply(4, 6);

export default function App() {
  return (
    <View style={styles.container}>
      <Text>Result: {result}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
