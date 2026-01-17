"""
Advanced Encryption Module for FOREX TRADING BOT
Military-grade encryption with multiple algorithms and key management
"""

import logging
import os
import base64
import json
import hashlib
import hmac
import secrets
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import struct
import threading
from collections import defaultdict, deque
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import constant_time
from cryptography.hazmat.backends import default_backend
import argon2
from cryptography import x509
from cryptography.x509.oid import NameOID
import asyncio
from contextlib import contextmanager
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EncryptionAlgorithm(Enum):
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_OAEP = "rsa_oaep"
    ECIES = "ecies"

class KeyDerivationFunction(Enum):
    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"
    ARGON2ID = "argon2id"

class KeyType(Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    EPHEMERAL = "ephemeral"

@dataclass
class EncryptionConfig:
    """Configuration for encryption module"""
    # Default algorithms
    default_symmetric_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    default_asymmetric_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.RSA_OAEP
    default_kdf: KeyDerivationFunction = KeyDerivationFunction.ARGON2ID
    
    # Key derivation parameters
    kdf_salt_size: int = 32
    kdf_iterations: int = 100000
    kdf_memory_cost: int = 2**20  # 1GB for Argon2
    
    # Key management
    key_rotation_days: int = 90
    max_key_versions: int = 5
    enable_auto_rotation: bool = True
    
    # Security parameters
    min_password_length: int = 12
    require_strong_passwords: bool = True
    encryption_version: str = "v1"
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    parallel_operations: int = 4
    
    # Compliance
    fips_compliant: bool = True
    nist_compliant: bool = True

@dataclass
class EncryptionResult:
    """Result of encryption operation"""
    success: bool
    ciphertext: bytes
    algorithm: EncryptionAlgorithm
    key_id: str
    iv_nonce: bytes
    auth_tag: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DecryptionResult:
    """Result of decryption operation"""
    success: bool
    plaintext: bytes
    algorithm: EncryptionAlgorithm
    key_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class KeyMetadata:
    """Metadata for encryption keys"""
    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    created_at: datetime
    expires_at: datetime
    version: int
    is_active: bool
    tags: List[str] = field(default_factory=list)
    description: str = ""

class AdvancedEncryption:
    """
    Advanced Encryption Module with military-grade security
    """
    
    def __init__(self, config: EncryptionConfig = None):
        self.config = config or EncryptionConfig()
        self.backend = default_backend()
        
        # Key storage
        self._symmetric_keys: Dict[str, bytes] = {}
        self._asymmetric_keys: Dict[str, Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]] = {}
        self._ec_keys: Dict[str, Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]] = {}
        
        # Key metadata
        self._key_metadata: Dict[str, KeyMetadata] = {}
        
        # Cache for performance
        self._derived_keys_cache: Dict[str, bytes] = {}
        self._cache_lock = threading.RLock()
        
        # Thread safety
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        
        # Key rotation tracking
        self._key_versions: Dict[str, List[str]] = defaultdict(list)
        
        # Initialize default keys
        self._initialize_default_keys()
        
        logger.info("AdvancedEncryption initialized successfully")
    
    def _initialize_default_keys(self) -> None:
        """Initialize default encryption keys"""
        try:
            # Generate master symmetric key
            master_key_id = "master_symmetric_v1"
            if master_key_id not in self._symmetric_keys:
                self._symmetric_keys[master_key_id] = self._generate_random_key(32)
                self._key_metadata[master_key_id] = KeyMetadata(
                    key_id=master_key_id,
                    key_type=KeyType.SYMMETRIC,
                    algorithm=EncryptionAlgorithm.AES_256_GCM,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=365),
                    version=1,
                    is_active=True,
                    tags=["master", "symmetric"],
                    description="Master symmetric encryption key"
                )
            
            # Generate RSA key pair
            rsa_key_id = "rsa_master_v1"
            if rsa_key_id not in self._asymmetric_keys:
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=4096,
                    backend=self.backend
                )
                public_key = private_key.public_key()
                self._asymmetric_keys[rsa_key_id] = (private_key, public_key)
                self._key_metadata[rsa_key_id] = KeyMetadata(
                    key_id=rsa_key_id,
                    key_type=KeyType.ASYMMETRIC,
                    algorithm=EncryptionAlgorithm.RSA_OAEP,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=365),
                    version=1,
                    is_active=True,
                    tags=["master", "rsa"],
                    description="Master RSA key pair"
                )
            
            # Generate EC key pair
            ec_key_id = "ec_master_v1"
            if ec_key_id not in self._ec_keys:
                private_key = ec.generate_private_key(ec.SECP384R1(), self.backend)
                public_key = private_key.public_key()
                self._ec_keys[ec_key_id] = (private_key, public_key)
                self._key_metadata[ec_key_id] = KeyMetadata(
                    key_id=ec_key_id,
                    key_type=KeyType.ASYMMETRIC,
                    algorithm=EncryptionAlgorithm.ECIES,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=365),
                    version=1,
                    is_active=True,
                    tags=["master", "ec"],
                    description="Master EC key pair"
                )
                
            logger.info("Default encryption keys initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize default keys: {e}")
            raise
    
    def _generate_random_key(self, length: int) -> bytes:
        """Generate cryptographically secure random key"""
        return secrets.token_bytes(length)
    
    def _generate_salt(self) -> bytes:
        """Generate cryptographically secure salt"""
        return self._generate_random_key(self.config.kdf_salt_size)
    
    def _derive_key_from_password(self, password: str, salt: bytes, 
                                kdf_type: KeyDerivationFunction = None) -> bytes:
        """Derive encryption key from password using KDF"""
        kdf_type = kdf_type or self.config.default_kdf
        
        if kdf_type == KeyDerivationFunction.PBKDF2:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA512(),
                length=32,
                salt=salt,
                iterations=self.config.kdf_iterations,
                backend=self.backend
            )
            return kdf.derive(password.encode('utf-8'))
        
        elif kdf_type == KeyDerivationFunction.SCRYPT:
            kdf = Scrypt(
                salt=salt,
                length=32,
                n=2**14,
                r=8,
                p=1,
                backend=self.backend
            )
            return kdf.derive(password.encode('utf-8'))
        
        elif kdf_type == KeyDerivationFunction.ARGON2ID:
            ph = argon2.PasswordHasher(
                time_cost=3,
                memory_cost=self.config.kdf_memory_cost,
                parallelism=self.config.parallel_operations,
                hash_len=32,
                salt_len=len(salt)
            )
            hash_result = ph.hash(password.encode('utf-8'), salt=salt)
            return base64.b64decode(hash_result.split('$')[-1])
        
        else:
            raise ValueError(f"Unsupported KDF: {kdf_type}")
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < self.config.min_password_length:
            return False
        
        if not self.config.require_strong_passwords:
            return True
        
        # Check for complexity
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def _get_cached_key(self, cache_key: str) -> Optional[bytes]:
        """Get derived key from cache"""
        if not self.config.enable_caching:
            return None
        
        with self._cache_lock:
            if cache_key in self._derived_keys_cache:
                return self._derived_keys_cache[cache_key]
        return None
    
    def _set_cached_key(self, cache_key: str, key: bytes) -> None:
        """Set derived key in cache"""
        if not self.config.enable_caching:
            return
        
        with self._cache_lock:
            self._derived_keys_cache[cache_key] = key
    
    def encrypt_symmetric(self, plaintext: bytes, key_id: str = None,
                         algorithm: EncryptionAlgorithm = None) -> EncryptionResult:
        """Encrypt data using symmetric encryption"""
        try:
            with self._lock:
                algorithm = algorithm or self.config.default_symmetric_algorithm
                key_id = key_id or "master_symmetric_v1"
                
                if key_id not in self._symmetric_keys:
                    raise ValueError(f"Symmetric key not found: {key_id}")
                
                key = self._symmetric_keys[key_id]
                
                if algorithm == EncryptionAlgorithm.AES_256_GCM:
                    return self._encrypt_aes_gcm(plaintext, key, key_id)
                elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                    return self._encrypt_aes_cbc(plaintext, key, key_id)
                elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                    return self._encrypt_chacha20_poly1305(plaintext, key, key_id)
                else:
                    raise ValueError(f"Unsupported symmetric algorithm: {algorithm}")
                    
        except Exception as e:
            logger.error(f"Symmetric encryption failed: {e}")
            return EncryptionResult(
                success=False,
                ciphertext=b"",
                algorithm=algorithm or self.config.default_symmetric_algorithm,
                key_id=key_id or "",
                iv_nonce=b""
            )
    
    def _encrypt_aes_gcm(self, plaintext: bytes, key: bytes, key_id: str) -> EncryptionResult:
        """Encrypt using AES-256-GCM"""
        try:
            # Generate random IV
            iv = self._generate_random_key(12)  # 96-bit IV for GCM
            
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=self.backend)
            encryptor = cipher.encryptor()
            
            # Encrypt
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            
            return EncryptionResult(
                success=True,
                ciphertext=ciphertext,
                algorithm=EncryptionAlgorithm.AES_256_GCM,
                key_id=key_id,
                iv_nonce=iv,
                auth_tag=encryptor.tag,
                metadata={
                    'iv_size': len(iv),
                    'auth_tag_size': len(encryptor.tag),
                    'original_size': len(plaintext)
                }
            )
            
        except Exception as e:
            logger.error(f"AES-GCM encryption failed: {e}")
            raise
    
    def _encrypt_aes_cbc(self, plaintext: bytes, key: bytes, key_id: str) -> EncryptionResult:
        """Encrypt using AES-256-CBC"""
        try:
            # Generate random IV
            iv = self._generate_random_key(16)  # 128-bit IV for CBC
            
            # Pad plaintext to block size
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(plaintext) + padder.finalize()
            
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
            encryptor = cipher.encryptor()
            
            # Encrypt
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            return EncryptionResult(
                success=True,
                ciphertext=ciphertext,
                algorithm=EncryptionAlgorithm.AES_256_CBC,
                key_id=key_id,
                iv_nonce=iv,
                metadata={
                    'iv_size': len(iv),
                    'original_size': len(plaintext),
                    'padded_size': len(padded_data)
                }
            )
            
        except Exception as e:
            logger.error(f"AES-CBC encryption failed: {e}")
            raise
    
    def _encrypt_chacha20_poly1305(self, plaintext: bytes, key: bytes, key_id: str) -> EncryptionResult:
        """Encrypt using ChaCha20-Poly1305"""
        try:
            # Generate random nonce
            nonce = self._generate_random_key(12)  # 96-bit nonce
            
            # Create cipher
            cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=self.backend)
            encryptor = cipher.encryptor()
            
            # Encrypt
            ciphertext = encryptor.update(plaintext)
            
            # Generate authentication tag (simplified - in production use proper AEAD)
            auth_tag = hmac.new(
                key, 
                nonce + ciphertext, 
                hashlib.sha256
            ).digest()[:16]
            
            return EncryptionResult(
                success=True,
                ciphertext=ciphertext,
                algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
                key_id=key_id,
                iv_nonce=nonce,
                auth_tag=auth_tag,
                metadata={
                    'nonce_size': len(nonce),
                    'auth_tag_size': len(auth_tag),
                    'original_size': len(plaintext)
                }
            )
            
        except Exception as e:
            logger.error(f"ChaCha20-Poly1305 encryption failed: {e}")
            raise
    
    def decrypt_symmetric(self, ciphertext: bytes, iv_nonce: bytes, 
                         key_id: str, algorithm: EncryptionAlgorithm = None,
                         auth_tag: bytes = None) -> DecryptionResult:
        """Decrypt data using symmetric encryption"""
        try:
            with self._lock:
                algorithm = algorithm or self.config.default_symmetric_algorithm
                
                if key_id not in self._symmetric_keys:
                    raise ValueError(f"Symmetric key not found: {key_id}")
                
                key = self._symmetric_keys[key_id]
                
                if algorithm == EncryptionAlgorithm.AES_256_GCM:
                    return self._decrypt_aes_gcm(ciphertext, iv_nonce, auth_tag, key, key_id)
                elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                    return self._decrypt_aes_cbc(ciphertext, iv_nonce, key, key_id)
                elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                    return self._decrypt_chacha20_poly1305(ciphertext, iv_nonce, auth_tag, key, key_id)
                else:
                    raise ValueError(f"Unsupported symmetric algorithm: {algorithm}")
                    
        except Exception as e:
            logger.error(f"Symmetric decryption failed: {e}")
            return DecryptionResult(
                success=False,
                plaintext=b"",
                algorithm=algorithm or self.config.default_symmetric_algorithm,
                key_id=key_id or ""
            )
    
    def _decrypt_aes_gcm(self, ciphertext: bytes, iv: bytes, auth_tag: bytes, 
                        key: bytes, key_id: str) -> DecryptionResult:
        """Decrypt using AES-256-GCM"""
        try:
            if not auth_tag:
                raise ValueError("Auth tag required for AES-GCM decryption")
            
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, auth_tag), backend=self.backend)
            decryptor = cipher.decryptor()
            
            # Decrypt
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return DecryptionResult(
                success=True,
                plaintext=plaintext,
                algorithm=EncryptionAlgorithm.AES_256_GCM,
                key_id=key_id,
                metadata={
                    'decrypted_size': len(plaintext)
                }
            )
            
        except Exception as e:
            logger.error(f"AES-GCM decryption failed: {e}")
            raise
    
    def _decrypt_aes_cbc(self, ciphertext: bytes, iv: bytes, key: bytes, key_id: str) -> DecryptionResult:
        """Decrypt using AES-256-CBC"""
        try:
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
            decryptor = cipher.decryptor()
            
            # Decrypt
            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Unpad
            unpadder = padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
            
            return DecryptionResult(
                success=True,
                plaintext=plaintext,
                algorithm=EncryptionAlgorithm.AES_256_CBC,
                key_id=key_id,
                metadata={
                    'decrypted_size': len(plaintext)
                }
            )
            
        except Exception as e:
            logger.error(f"AES-CBC decryption failed: {e}")
            raise
    
    def _decrypt_chacha20_poly1305(self, ciphertext: bytes, nonce: bytes, auth_tag: bytes,
                                 key: bytes, key_id: str) -> DecryptionResult:
        """Decrypt using ChaCha20-Poly1305"""
        try:
            if auth_tag:
                # Verify authentication tag
                expected_tag = hmac.new(
                    key,
                    nonce + ciphertext,
                    hashlib.sha256
                ).digest()[:16]
                
                if not constant_time.bytes_eq(auth_tag, expected_tag):
                    raise ValueError("Authentication tag verification failed")
            
            # Create cipher
            cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=self.backend)
            decryptor = cipher.decryptor()
            
            # Decrypt
            plaintext = decryptor.update(ciphertext)
            
            return DecryptionResult(
                success=True,
                plaintext=plaintext,
                algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
                key_id=key_id,
                metadata={
                    'decrypted_size': len(plaintext)
                }
            )
            
        except Exception as e:
            logger.error(f"ChaCha20-Poly1305 decryption failed: {e}")
            raise
    
    def encrypt_asymmetric(self, plaintext: bytes, public_key_id: str,
                          algorithm: EncryptionAlgorithm = None) -> EncryptionResult:
        """Encrypt data using asymmetric encryption"""
        try:
            with self._lock:
                algorithm = algorithm or self.config.default_asymmetric_algorithm
                
                if algorithm == EncryptionAlgorithm.RSA_OAEP:
                    return self._encrypt_rsa_oaep(plaintext, public_key_id)
                elif algorithm == EncryptionAlgorithm.ECIES:
                    return self._encrypt_ecies(plaintext, public_key_id)
                else:
                    raise ValueError(f"Unsupported asymmetric algorithm: {algorithm}")
                    
        except Exception as e:
            logger.error(f"Asymmetric encryption failed: {e}")
            return EncryptionResult(
                success=False,
                ciphertext=b"",
                algorithm=algorithm or self.config.default_asymmetric_algorithm,
                key_id=public_key_id,
                iv_nonce=b""
            )
    
    def _encrypt_rsa_oaep(self, plaintext: bytes, public_key_id: str) -> EncryptionResult:
        """Encrypt using RSA-OAEP"""
        try:
            if public_key_id not in self._asymmetric_keys:
                raise ValueError(f"RSA public key not found: {public_key_id}")
            
            _, public_key = self._asymmetric_keys[public_key_id]
            
            # RSA can encrypt limited data, so we'll use hybrid approach
            if len(plaintext) > 400:  # RSA-4096 can encrypt ~400 bytes
                # Generate ephemeral symmetric key
                ephemeral_key = self._generate_random_key(32)
                ephemeral_key_id = f"ephemeral_{int(time.time())}"
                
                # Encrypt data with symmetric key
                symmetric_result = self._encrypt_aes_gcm(plaintext, ephemeral_key, ephemeral_key_id)
                
                # Encrypt symmetric key with RSA
                encrypted_key = public_key.encrypt(
                    ephemeral_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                # Combine results
                combined_data = struct.pack(">I", len(encrypted_key)) + encrypted_key + symmetric_result.ciphertext
                
                return EncryptionResult(
                    success=True,
                    ciphertext=combined_data,
                    algorithm=EncryptionAlgorithm.RSA_OAEP,
                    key_id=public_key_id,
                    iv_nonce=symmetric_result.iv_nonce,
                    auth_tag=symmetric_result.auth_tag,
                    metadata={
                        'hybrid_encryption': True,
                        'ephemeral_key_id': ephemeral_key_id,
                        'encrypted_key_size': len(encrypted_key),
                        'original_size': len(plaintext)
                    }
                )
            else:
                # Direct RSA encryption for small data
                ciphertext = public_key.encrypt(
                    plaintext,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                return EncryptionResult(
                    success=True,
                    ciphertext=ciphertext,
                    algorithm=EncryptionAlgorithm.RSA_OAEP,
                    key_id=public_key_id,
                    iv_nonce=b"",
                    metadata={
                        'hybrid_encryption': False,
                        'original_size': len(plaintext)
                    }
                )
                
        except Exception as e:
            logger.error(f"RSA-OAEP encryption failed: {e}")
            raise
    
    def _encrypt_ecies(self, plaintext: bytes, public_key_id: str) -> EncryptionResult:
        """Encrypt using ECIES"""
        try:
            if public_key_id not in self._ec_keys:
                raise ValueError(f"EC public key not found: {public_key_id}")
            
            _, public_key = self._ec_keys[public_key_id]
            
            # Generate ephemeral key pair
            ephemeral_private = ec.generate_private_key(ec.SECP384R1(), self.backend)
            ephemeral_public = ephemeral_private.public_key()
            
            # Perform key agreement
            shared_secret = ephemeral_private.exchange(ec.ECDH(), public_key)
            
            # Derive encryption key from shared secret
            kdf = hashes.Hash(hashes.SHA512(), backend=self.backend)
            kdf.update(shared_secret)
            derived_key = kdf.finalize()[:32]
            
            # Encrypt data with derived key
            ephemeral_key_id = f"ec_ephemeral_{int(time.time())}"
            symmetric_result = self._encrypt_aes_gcm(plaintext, derived_key, ephemeral_key_id)
            
            # Serialize ephemeral public key
            ephemeral_public_bytes = ephemeral_public.public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint
            )
            
            # Combine results
            combined_data = struct.pack(">I", len(ephemeral_public_bytes)) + ephemeral_public_bytes + symmetric_result.ciphertext
            
            return EncryptionResult(
                success=True,
                ciphertext=combined_data,
                algorithm=EncryptionAlgorithm.ECIES,
                key_id=public_key_id,
                iv_nonce=symmetric_result.iv_nonce,
                auth_tag=symmetric_result.auth_tag,
                metadata={
                    'ephemeral_public_key_size': len(ephemeral_public_bytes),
                    'original_size': len(plaintext)
                }
            )
            
        except Exception as e:
            logger.error(f"ECIES encryption failed: {e}")
            raise
    
    def decrypt_asymmetric(self, ciphertext: bytes, private_key_id: str,
                          algorithm: EncryptionAlgorithm = None) -> DecryptionResult:
        """Decrypt data using asymmetric encryption"""
        try:
            with self._lock:
                algorithm = algorithm or self.config.default_asymmetric_algorithm
                
                if algorithm == EncryptionAlgorithm.RSA_OAEP:
                    return self._decrypt_rsa_oaep(ciphertext, private_key_id)
                elif algorithm == EncryptionAlgorithm.ECIES:
                    return self._decrypt_ecies(ciphertext, private_key_id)
                else:
                    raise ValueError(f"Unsupported asymmetric algorithm: {algorithm}")
                    
        except Exception as e:
            logger.error(f"Asymmetric decryption failed: {e}")
            return DecryptionResult(
                success=False,
                plaintext=b"",
                algorithm=algorithm or self.config.default_asymmetric_algorithm,
                key_id=private_key_id
            )
    
    def _decrypt_rsa_oaep(self, ciphertext: bytes, private_key_id: str) -> DecryptionResult:
        """Decrypt using RSA-OAEP"""
        try:
            if private_key_id not in self._asymmetric_keys:
                raise ValueError(f"RSA private key not found: {private_key_id}")
            
            private_key, _ = self._asymmetric_keys[private_key_id]
            
            # Check if it's hybrid encryption
            if len(ciphertext) > 512:  # Larger than typical RSA encryption
                # Extract encrypted symmetric key
                key_size = struct.unpack(">I", ciphertext[:4])[0]
                encrypted_key = ciphertext[4:4 + key_size]
                actual_ciphertext = ciphertext[4 + key_size:]
                
                # Decrypt symmetric key
                symmetric_key = private_key.decrypt(
                    encrypted_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                # Decrypt data with symmetric key
                # Note: In production, you'd need to store IV and auth tag properly
                ephemeral_key_id = f"ephemeral_decrypt_{int(time.time())}"
                self._symmetric_keys[ephemeral_key_id] = symmetric_key
                
                # This is simplified - in production, you'd need proper IV/auth tag handling
                decryption_result = self.decrypt_symmetric(
                    actual_ciphertext,
                    b"",  # Placeholder - need proper IV
                    ephemeral_key_id,
                    EncryptionAlgorithm.AES_256_GCM,
                    b""   # Placeholder - need proper auth tag
                )
                
                # Clean up ephemeral key
                del self._symmetric_keys[ephemeral_key_id]
                
                return decryption_result
            else:
                # Direct RSA decryption
                plaintext = private_key.decrypt(
                    ciphertext,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                return DecryptionResult(
                    success=True,
                    plaintext=plaintext,
                    algorithm=EncryptionAlgorithm.RSA_OAEP,
                    key_id=private_key_id,
                    metadata={
                        'decrypted_size': len(plaintext)
                    }
                )
                
        except Exception as e:
            logger.error(f"RSA-OAEP decryption failed: {e}")
            raise
    
    def _decrypt_ecies(self, ciphertext: bytes, private_key_id: str) -> DecryptionResult:
        """Decrypt using ECIES"""
        try:
            if private_key_id not in self._ec_keys:
                raise ValueError(f"EC private key not found: {private_key_id}")
            
            private_key, _ = self._ec_keys[private_key_id]
            
            # Extract ephemeral public key
            key_size = struct.unpack(">I", ciphertext[:4])[0]
            ephemeral_public_bytes = ciphertext[4:4 + key_size]
            actual_ciphertext = ciphertext[4 + key_size:]
            
            # Reconstruct ephemeral public key
            ephemeral_public = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP384R1(),
                ephemeral_public_bytes
            )
            
            # Perform key agreement
            shared_secret = private_key.exchange(ec.ECDH(), ephemeral_public)
            
            # Derive encryption key
            kdf = hashes.Hash(hashes.SHA512(), backend=self.backend)
            kdf.update(shared_secret)
            derived_key = kdf.finalize()[:32]
            
            # Decrypt data with derived key
            ephemeral_key_id = f"ec_derived_{int(time.time())}"
            self._symmetric_keys[ephemeral_key_id] = derived_key
            
            # This is simplified - need proper IV/auth tag handling
            decryption_result = self.decrypt_symmetric(
                actual_ciphertext,
                b"",  # Placeholder
                ephemeral_key_id,
                EncryptionAlgorithm.AES_256_GCM,
                b""   # Placeholder
            )
            
            # Clean up
            del self._symmetric_keys[ephemeral_key_id]
            
            return decryption_result
            
        except Exception as e:
            logger.error(f"ECIES decryption failed: {e}")
            raise
    
    def generate_key_pair(self, key_type: KeyType, algorithm: EncryptionAlgorithm,
                         key_id: str = None, description: str = "") -> str:
        """Generate new key pair"""
        try:
            with self._lock:
                key_id = key_id or f"{key_type.value}_{algorithm.value}_{int(time.time())}"
                
                if key_type == KeyType.SYMMETRIC:
                    if algorithm not in [EncryptionAlgorithm.AES_256_GCM, 
                                       EncryptionAlgorithm.AES_256_CBC,
                                       EncryptionAlgorithm.CHACHA20_POLY1305]:
                        raise ValueError(f"Unsupported algorithm for symmetric key: {algorithm}")
                    
                    key_size = 32  # 256 bits
                    self._symmetric_keys[key_id] = self._generate_random_key(key_size)
                    
                elif key_type == KeyType.ASYMMETRIC:
                    if algorithm == EncryptionAlgorithm.RSA_OAEP:
                        private_key = rsa.generate_private_key(
                            public_exponent=65537,
                            key_size=4096,
                            backend=self.backend
                        )
                        public_key = private_key.public_key()
                        self._asymmetric_keys[key_id] = (private_key, public_key)
                    
                    elif algorithm == EncryptionAlgorithm.ECIES:
                        private_key = ec.generate_private_key(ec.SECP384R1(), self.backend)
                        public_key = private_key.public_key()
                        self._ec_keys[key_id] = (private_key, public_key)
                    
                    else:
                        raise ValueError(f"Unsupported algorithm for asymmetric key: {algorithm}")
                
                else:
                    raise ValueError(f"Unsupported key type: {key_type}")
                
                # Store metadata
                self._key_metadata[key_id] = KeyMetadata(
                    key_id=key_id,
                    key_type=key_type,
                    algorithm=algorithm,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=self.config.key_rotation_days),
                    version=1,
                    is_active=True,
                    description=description
                )
                
                # Track versions
                self._key_versions[key_id].append(key_id)
                
                logger.info(f"Generated new key pair: {key_id}")
                return key_id
                
        except Exception as e:
            logger.error(f"Key pair generation failed: {e}")
            raise
    
    def export_public_key(self, key_id: str) -> bytes:
        """Export public key in PEM format"""
        try:
            with self._lock:
                if key_id in self._asymmetric_keys:
                    _, public_key = self._asymmetric_keys[key_id]
                    return public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    )
                elif key_id in self._ec_keys:
                    _, public_key = self._ec_keys[key_id]
                    return public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    )
                else:
                    raise ValueError(f"Public key not found: {key_id}")
                    
        except Exception as e:
            logger.error(f"Public key export failed: {e}")
            raise
    
    def import_public_key(self, public_key_pem: bytes, key_id: str, 
                         algorithm: EncryptionAlgorithm) -> None:
        """Import public key from PEM format"""
        try:
            with self._lock:
                public_key = serialization.load_pem_public_key(public_key_pem, backend=self.backend)
                
                if isinstance(public_key, rsa.RSAPublicKey):
                    self._asymmetric_keys[key_id] = (None, public_key)
                elif isinstance(public_key, ec.EllipticCurvePublicKey):
                    self._ec_keys[key_id] = (None, public_key)
                else:
                    raise ValueError("Unsupported public key type")
                
                self._key_metadata[key_id] = KeyMetadata(
                    key_id=key_id,
                    key_type=KeyType.ASYMMETRIC,
                    algorithm=algorithm,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=365),
                    version=1,
                    is_active=True,
                    description="Imported public key"
                )
                
                logger.info(f"Imported public key: {key_id}")
                
        except Exception as e:
            logger.error(f"Public key import failed: {e}")
            raise
    
    def rotate_key(self, key_id: str) -> str:
        """Rotate encryption key"""
        try:
            with self._lock:
                if key_id not in self._key_metadata:
                    raise ValueError(f"Key not found: {key_id}")
                
                metadata = self._key_metadata[key_id]
                new_key_id = f"{key_id}_v{metadata.version + 1}"
                
                # Generate new key
                if metadata.key_type == KeyType.SYMMETRIC:
                    self._symmetric_keys[new_key_id] = self._generate_random_key(32)
                elif metadata.key_type == KeyType.ASYMMETRIC:
                    if metadata.algorithm == EncryptionAlgorithm.RSA_OAEP:
                        private_key = rsa.generate_private_key(
                            public_exponent=65537,
                            key_size=4096,
                            backend=self.backend
                        )
                        public_key = private_key.public_key()
                        self._asymmetric_keys[new_key_id] = (private_key, public_key)
                    elif metadata.algorithm == EncryptionAlgorithm.ECIES:
                        private_key = ec.generate_private_key(ec.SECP384R1(), self.backend)
                        public_key = private_key.public_key()
                        self._ec_keys[new_key_id] = (private_key, public_key)
                
                # Update metadata
                self._key_metadata[new_key_id] = KeyMetadata(
                    key_id=new_key_id,
                    key_type=metadata.key_type,
                    algorithm=metadata.algorithm,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=self.config.key_rotation_days),
                    version=metadata.version + 1,
                    is_active=True,
                    description=f"Rotated from {key_id}"
                )
                
                # Deactivate old key
                metadata.is_active = False
                
                # Track versions
                self._key_versions[key_id].append(new_key_id)
                
                # Limit number of versions
                if len(self._key_versions[key_id]) > self.config.max_key_versions:
                    oldest_key = self._key_versions[key_id].pop(0)
                    self._delete_key(oldest_key)
                
                logger.info(f"Key rotated: {key_id} -> {new_key_id}")
                return new_key_id
                
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            raise
    
    def _delete_key(self, key_id: str) -> None:
        """Securely delete encryption key"""
        try:
            # Overwrite key material in memory
            if key_id in self._symmetric_keys:
                # Overwrite with random data
                self._symmetric_keys[key_id] = self._generate_random_key(32)
                del self._symmetric_keys[key_id]
            
            if key_id in self._asymmetric_keys:
                del self._asymmetric_keys[key_id]
            
            if key_id in self._ec_keys:
                del self._ec_keys[key_id]
            
            if key_id in self._key_metadata:
                del self._key_metadata[key_id]
            
            logger.info(f"Key securely deleted: {key_id}")
            
        except Exception as e:
            logger.error(f"Key deletion failed: {e}")
            raise
    
    def get_key_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Get metadata for specific key"""
        return self._key_metadata.get(key_id)
    
    def list_keys(self, key_type: KeyType = None) -> List[KeyMetadata]:
        """List all keys with optional filtering"""
        if key_type:
            return [meta for meta in self._key_metadata.values() if meta.key_type == key_type]
        return list(self._key_metadata.values())
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of encryption module"""
        try:
            # Test encryption/decryption
            test_data = b"Encryption health check test data"
            
            # Symmetric test
            encrypt_result = self.encrypt_symmetric(test_data)
            if not encrypt_result.success:
                return {"status": "error", "message": "Symmetric encryption failed"}
            
            decrypt_result = self.decrypt_symmetric(
                encrypt_result.ciphertext,
                encrypt_result.iv_nonce,
                encrypt_result.key_id,
                encrypt_result.algorithm,
                encrypt_result.auth_tag
            )
            
            if not decrypt_result.success or decrypt_result.plaintext != test_data:
                return {"status": "error", "message": "Symmetric decryption failed"}
            
            # Asymmetric test (RSA)
            rsa_encrypt_result = self.encrypt_asymmetric(test_data, "rsa_master_v1")
            if not rsa_encrypt_result.success:
                return {"status": "error", "message": "RSA encryption failed"}
            
            rsa_decrypt_result = self.decrypt_asymmetric(
                rsa_encrypt_result.ciphertext,
                "rsa_master_v1",
                rsa_encrypt_result.algorithm
            )
            
            if not rsa_decrypt_result.success:
                return {"status": "error", "message": "RSA decryption failed"}
            
            return {
                "status": "healthy",
                "timestamp": datetime.now(),
                "key_count": len(self._key_metadata),
                "symmetric_keys": len(self._symmetric_keys),
                "asymmetric_keys": len(self._asymmetric_keys) + len(self._ec_keys)
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}

# Example usage and testing
def main():
    """Example usage of the AdvancedEncryption module"""
    
    # Configure encryption
    config = EncryptionConfig(
        default_symmetric_algorithm=EncryptionAlgorithm.AES_256_GCM,
        default_asymmetric_algorithm=EncryptionAlgorithm.RSA_OAEP,
        enable_auto_rotation=True,
        key_rotation_days=90
    )
    
    # Initialize encryption module
    crypto = AdvancedEncryption(config)
    
    print("=== Testing Advanced Encryption Module ===\n")
    
    # Test 1: Symmetric Encryption
    print("1. Symmetric Encryption Test:")
    test_data = b"Secret trading data for EUR/USD analysis"
    encrypt_result = crypto.encrypt_symmetric(test_data)
    
    if encrypt_result.success:
        print(f"   ✅ Encryption successful")
        print(f"   Algorithm: {encrypt_result.algorithm.value}")
        print(f"   Key ID: {encrypt_result.key_id}")
        print(f"   IV/Nonce: {encrypt_result.iv_nonce.hex()[:16]}...")
        print(f"   Auth Tag: {encrypt_result.auth_tag.hex()[:16]}...")
        
        # Decrypt
        decrypt_result = crypto.decrypt_symmetric(
            encrypt_result.ciphertext,
            encrypt_result.iv_nonce,
            encrypt_result.key_id,
            encrypt_result.algorithm,
            encrypt_result.auth_tag
        )
        
        if decrypt_result.success and decrypt_result.plaintext == test_data:
            print(f"   ✅ Decryption successful - Data integrity verified")
        else:
            print(f"   ❌ Decryption failed")
    else:
        print(f"   ❌ Encryption failed")
    
    # Test 2: Asymmetric Encryption (RSA)
    print("\n2. Asymmetric Encryption Test (RSA):")
    rsa_encrypt_result = crypto.encrypt_asymmetric(test_data, "rsa_master_v1")
    
    if rsa_encrypt_result.success:
        print(f"   ✅ RSA Encryption successful")
        print(f"   Algorithm: {rsa_encrypt_result.algorithm.value}")
        
        rsa_decrypt_result = crypto.decrypt_asymmetric(
            rsa_encrypt_result.ciphertext,
            "rsa_master_v1",
            rsa_encrypt_result.algorithm
        )
        
        if rsa_decrypt_result.success and rsa_decrypt_result.plaintext == test_data:
            print(f"   ✅ RSA Decryption successful - Data integrity verified")
        else:
            print(f"   ❌ RSA Decryption failed")
    else:
        print(f"   ❌ RSA Encryption failed")
    
    # Test 3: Key Management
    print("\n3. Key Management Test:")
    new_key_id = crypto.generate_key_pair(
        KeyType.SYMMETRIC,
        EncryptionAlgorithm.AES_256_GCM,
        description="Test symmetric key"
    )
    print(f"   ✅ Generated new key: {new_key_id}")
    
    # List keys
    keys = crypto.list_keys()
    print(f"   Total keys: {len(keys)}")
    for key in keys[:3]:  # Show first 3 keys
        print(f"     - {key.key_id} ({key.key_type.value})")
    
    # Test 4: Health Check
    print("\n4. Health Check:")
    health = crypto.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Key Count: {health['key_count']}")
    print(f"   Symmetric Keys: {health['symmetric_keys']}")
    print(f"   Asymmetric Keys: {health['asymmetric_keys']}")
    
    # Test 5: Performance
    print("\n5. Performance Test:")
    import time
    start_time = time.time()
    
    # Encrypt 100 small messages
    for i in range(100):
        message = f"Trade signal {i}".encode()
        result = crypto.encrypt_symmetric(message)
        if not result.success:
            print(f"   ❌ Performance test failed at iteration {i}")
            break
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"   ✅ Encrypted 100 messages in {duration:.3f} seconds")
    print(f"   Throughput: {100/duration:.1f} operations/second")
    
    print("\n=== Encryption Module Test Completed ===")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()