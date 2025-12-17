// IndexedDB Setup and CRUD Operations

let db;
const DB_NAME = 'FashionSenseDB';
const DB_VERSION = 2; // Increment when modifying DB schema
const WARDROBE_STORE = 'wardrobe';
const HISTORY_STORE = 'history';

export function getDb() {
    return db;
}

export function initDB(retryCount = 0) {
    return new Promise((resolve, reject) => {
        console.log('[initDB] opening IndexedDB...');
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onerror = (event) => {
            console.error('[initDB] ERROR:', event.target.error);
            
            // If first attempt failed, try deleting and recreating the database
            if (retryCount === 0) {
                console.log('[initDB] Attempting to delete and recreate database...');
                const deleteRequest = indexedDB.deleteDatabase(DB_NAME);
                deleteRequest.onsuccess = () => {
                    console.log('[initDB] Database deleted, retrying...');
                    initDB(1).then(resolve).catch(reject);
                };
                deleteRequest.onerror = () => {
                    reject(event.target.error || new Error('IndexedDB error'));
                };
            } else {
                reject(event.target.error || new Error('IndexedDB error'));
            }
        };

        request.onblocked = () => {
            console.error('[initDB] BLOCKED — please close other tabs using this app and refresh');
        };

        request.onsuccess = (event) => {
            db = event.target.result;
            console.log('[initDB] IndexedDB initialized');
            resolve(db);
        };

        request.onupgradeneeded = (event) => {
            console.log('[initDB] upgrade needed');
            const database = event.target.result;
            db = database;

            if (!database.objectStoreNames.contains(WARDROBE_STORE)) {
                console.log('[initDB] creating wardrobe store');
                database.createObjectStore(WARDROBE_STORE, {
                    keyPath: 'id',
                    autoIncrement: true
                });
            }

            if (!database.objectStoreNames.contains(HISTORY_STORE)) {
                console.log('[initDB] creating history store');
                const historyStore = database.createObjectStore(HISTORY_STORE, {
                    keyPath: 'timestamp'
                });
                historyStore.createIndex('timestamp', 'timestamp', { unique: true });
            }
        };
    });
}

// Wardrobe CRUD operations
export async function addToWardrobe(imageDataURL, fileName) {
    if (!db) throw new Error('Wardrobe database is not initialized');

    return new Promise((resolve, reject) => {
        try {
            const transaction = db.transaction([WARDROBE_STORE], 'readwrite');
            const store = transaction.objectStore(WARDROBE_STORE);

            const item = {
                imageData: imageDataURL,
                fileName: fileName || 'wardrobe-item.jpg',
                timestamp: new Date().toISOString()
            };

            const request = store.add(item);

            request.onsuccess = () => resolve(request.result);
            request.onerror  = (event) => {
                const error = event.target.error;
                // Check for quota exceeded
                if (error.name === 'QuotaExceededError') {
                    const quotaError = new Error('Storage quota exceeded. Try clearing some wardrobe items or history.');
                    quotaError.name = 'QuotaExceededError';
                    reject(quotaError);
                } else {
                    reject(error);
                }
            };
        } catch (err) {
            reject(err);
        }
    });
}

export async function getAllWardrobeItems() {
    if (!db) throw new Error('Wardrobe database is not initialized');

    return new Promise((resolve, reject) => {
        try {
            const transaction = db.transaction([WARDROBE_STORE], 'readonly');
            const store = transaction.objectStore(WARDROBE_STORE);
            const request = store.getAll();

            request.onsuccess = () => resolve(request.result);
            request.onerror  = () => reject(request.error);
        } catch (err) {
            reject(err);
        }
    });
}

export async function deleteWardrobeItem(id) {
    if (!db) throw new Error('Wardrobe database is not initialized');

    return new Promise((resolve, reject) => {
        try {
            const transaction = db.transaction([WARDROBE_STORE], 'readwrite');
            const store = transaction.objectStore(WARDROBE_STORE);
            const request = store.delete(id);

            request.onsuccess = () => resolve();
            request.onerror  = () => reject(request.error);
        } catch (err) {
            reject(err);
        }
    });
}

// History CRUD operations (using IndexedDB for large storage capacity)
export async function addToHistoryDB(result) {
    if (!db) throw new Error('Database is not initialized');

    return new Promise((resolve, reject) => {
        try {
            const transaction = db.transaction([HISTORY_STORE], 'readwrite');
            const store = transaction.objectStore(HISTORY_STORE);

            const historyItem = {
                context: result.context,
                attributes: result.attributes,
                recommendation: result.recommendation,
                selectedItems: result.selectedItems,
                images: result.images || [],
                timestamp: result.timestamp
            };

            const request = store.put(historyItem); // put will update if exists, add if not

            request.onsuccess = () => resolve(true);
            request.onerror = (event) => {
                const error = event.target.error;
                // Check for quota exceeded
                if (error.name === 'QuotaExceededError') {
                    const quotaError = new Error('Storage quota exceeded. Try clearing history or using fewer/smaller images.');
                    quotaError.name = 'QuotaExceededError';
                    reject(quotaError);
                } else {
                    reject(error);
                }
            };
        } catch (err) {
            reject(err);
        }
    });
}

export async function getAllHistoryDB() {
    if (!db) throw new Error('Database is not initialized');

    return new Promise((resolve, reject) => {
        try {
            const transaction = db.transaction([HISTORY_STORE], 'readonly');
            const store = transaction.objectStore(HISTORY_STORE);
            const request = store.getAll();

            request.onsuccess = () => {
                // Sort by timestamp descending (newest first)
                const results = request.result.sort((a, b) => 
                    new Date(b.timestamp) - new Date(a.timestamp)
                );
                resolve(results);
            };
            request.onerror = () => reject(request.error);
        } catch (err) {
            reject(err);
        }
    });
}

export async function getHistoryItemDB(timestamp) {
    if (!db) throw new Error('Database is not initialized');

    return new Promise((resolve, reject) => {
        try {
            const transaction = db.transaction([HISTORY_STORE], 'readonly');
            const store = transaction.objectStore(HISTORY_STORE);
            const request = store.get(timestamp);

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        } catch (err) {
            reject(err);
        }
    });
}

export async function deleteFromHistoryDB(timestamp) {
    if (!db) throw new Error('Database is not initialized');

    return new Promise((resolve, reject) => {
        try {
            const transaction = db.transaction([HISTORY_STORE], 'readwrite');
            const store = transaction.objectStore(HISTORY_STORE);
            const request = store.delete(timestamp);

            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        } catch (err) {
            reject(err);
        }
    });
}

export async function clearAllHistoryDB() {
    if (!db) throw new Error('Database is not initialized');

    return new Promise((resolve, reject) => {
        try {
            const transaction = db.transaction([HISTORY_STORE], 'readwrite');
            const store = transaction.objectStore(HISTORY_STORE);
            const request = store.clear();

            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        } catch (err) {
            reject(err);
        }
    });
}

// Migrate existing localStorage history to IndexedDB (one-time migration)
export async function migrateHistoryToIndexedDB() {
    const oldHistory = localStorage.getItem('fashionSenseHistory');
    if (oldHistory) {
        try {
            const historyItems = JSON.parse(oldHistory);
            console.log('[Migration] Found', historyItems.length, 'history items in localStorage');
            
            for (const item of historyItems) {
                await addToHistoryDB(item);
            }
            
            // Clear localStorage after successful migration
            localStorage.removeItem('fashionSenseHistory');
            console.log('[Migration] Successfully migrated history to IndexedDB');
        } catch (err) {
            console.error('[Migration] Failed to migrate history:', err);
        }
    }
}

// Save wardrobe order to IndexedDB
export async function saveWardrobeOrder(orderedIds) {
    if (!db) return;
    
    try {
        const transaction = db.transaction([WARDROBE_STORE], 'readwrite');
        const store = transaction.objectStore(WARDROBE_STORE);
        
        // Update each item with its new order
        for (let i = 0; i < orderedIds.length; i++) {
            const id = orderedIds[i];
            const getRequest = store.get(id);
            
            await new Promise((resolve, reject) => {
                getRequest.onsuccess = () => {
                    const item = getRequest.result;
                    if (item) {
                        item.order = i;
                        const putRequest = store.put(item);
                        putRequest.onsuccess = resolve;
                        putRequest.onerror = reject;
                    } else {
                        resolve();
                    }
                };
                getRequest.onerror = reject;
            });
        }
    } catch (error) {
        console.error('Error saving wardrobe order:', error);
    }
}
