/**
 * Copyright (c) 2018 - 2019, Nordic Semiconductor ASA
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form, except as embedded into a Nordic
 *    Semiconductor ASA integrated circuit in a product or a software update for
 *    such product, must reproduce the above copyright notice, this list of
 *    conditions and the following disclaimer in the documentation and/or other
 *    materials provided with the distribution.
 *
 * 3. Neither the name of Nordic Semiconductor ASA nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * 4. This software, with or without modification, must only be used with a
 *    Nordic Semiconductor ASA integrated circuit.
 *
 * 5. Any software provided in binary form under this license must not be reverse
 *    engineered, decompiled, modified and/or disassembled.
 *
 * THIS SOFTWARE IS PROVIDED BY NORDIC SEMICONDUCTOR ASA "AS IS" AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NORDIC SEMICONDUCTOR ASA OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "nfc_m.h"
#include "sdk_errors.h"
#include "nfc_t2t_lib.h"
#include "nrf_drv_rng.h"
#include "nfc_pair_lib_m.h"

#define TK_MAX_NUM                      1                               /**< Maximal number of TK locations in NDEF message buffer. */
#define NDEF_MSG_BUFF_SIZE              256                             /**< Size of buffer for the NDEF pairing message. */

#define BLE_NFC_SEC_PARAM_KEYPRESS      0                               /**< Keypress notifications not enabled. */
#define BLE_NFC_SEC_PARAM_IO_CAPS       BLE_GAP_IO_CAPS_NONE            /**< No I/O capabilities. */

static ble_advertising_t *       m_p_advertising = NULL;                /**< Pointer to the advertising module instance. */

static uint8_t                   m_ndef_msg_buf[NDEF_MSG_BUFF_SIZE];    /**< NFC tag NDEF message buffer. */
static ble_advdata_tk_value_t    m_oob_auth_key;                        /**< Temporary Key buffer used in OOB legacy pairing mode. */
static uint8_t *                 m_tk_group[TK_MAX_NUM];                /**< Locations of TK in NDEF message. */
//static nfc_pairing_mode_t        m_pairing_mode;                        /**< Current pairing mode. */
static ble_gap_sec_params_t      m_sec_param;                           /**< Current Peer Manager secure parameters configuration. */

static uint8_t                   m_connections = 0;                     /**< Number of active connections. */

static void ble_evt_handler(ble_evt_t const * p_ble_evt, void * p_context);

//NRF_SDH_BLE_OBSERVER(m_ble_evt_observer, NFC_BLE_PAIR_LIB_BLE_OBSERVER_PRIO, ble_evt_handler, NULL);

typedef enum
{
    NFC_PAIRING_MODE_JUST_WORKS,        /**< Legacy Just Works pairing without a security key. */
    NFC_PAIRING_MODE_OOB,               /**< Legacy OOB pairing with a Temporary Key shared through NFC tag data. */
    NFC_PAIRING_MODE_LESC_JUST_WORKS,   /**< LESC pairing without authentication data. */
    NFC_PAIRING_MODE_LESC_OOB,          /**< LESC pairing with OOB authentication data. */
    NFC_PAIRING_MODE_GENERIC_OOB,       /**< OOB pairing with fallback from LESC to Legacy mode. */
    NFC_PAIRING_MODE_CNT                /**< Number of available pairing modes. */
} nfc_pairing_mode_t;


static void nfc_callback(void            * p_context,
                         nfc_t2t_event_t   event,
                         uint8_t const   * p_data,
                         size_t            data_length)
{
    UNUSED_PARAMETER(p_context);
    UNUSED_PARAMETER(p_data);
    UNUSED_PARAMETER(data_length);

    ret_code_t         err_code = NRF_SUCCESS;
    nfc_pairing_mode_t pairing_mode;

    switch (event)
    {
        case NFC_T2T_EVENT_FIELD_ON:

            pairing_mode = nfc_ble_pair_mode_get();

            if ((pairing_mode == NFC_PAIRING_MODE_OOB) ||
                (pairing_mode == NFC_PAIRING_MODE_GENERIC_OOB))
            {
                // Generate Authentication OOB Key and update NDEF message content.
                uint8_t length = random_vector_generate(m_oob_auth_key.tk, BLE_GAP_SEC_KEY_LEN);
                random_vector_log(length);
                err_code = nfc_tk_group_modifier_update(&m_oob_auth_key);
                APP_ERROR_CHECK(err_code);
            }

            // Start advertising when NFC field is sensed and there is a place for another connection.
            if (m_connections < NRF_SDH_BLE_PERIPHERAL_LINK_COUNT)
            {
                err_code = ble_advertising_start(m_p_advertising, BLE_ADV_MODE_FAST);
                if (err_code != NRF_ERROR_INVALID_STATE)
                {
                    APP_ERROR_CHECK(err_code);
                }
            }

            break;

        case NFC_T2T_EVENT_FIELD_OFF:
            NRF_LOG_DEBUG("NFC_EVENT_FIELD_OFF");
            break;

        default:
            break;
    }
}


ret_code_t nfc_ble_pair_init(void)
{
    ret_code_t err_code = NRF_SUCCESS;

    // Initialize RNG peripheral for authentication OOB data generation
    err_code = nrf_drv_rng_init(NULL);
    if (err_code != NRF_ERROR_INVALID_STATE &&
        err_code != NRF_ERROR_MODULE_ALREADY_INITIALIZED)
    {
        VERIFY_SUCCESS(err_code);
    }

    // Start NFC.
    err_code = nfc_t2t_setup(nfc_callback, NULL);
    VERIFY_SUCCESS(err_code);

    // Set proper NFC data.
    err_code = nfc_ble_pair_data_set();
    VERIFY_SUCCESS(err_code);

    return err_code;
}


void nfc_pairing_init(void)
{
    ret_code_t err_code = nfc_ble_pair_init();

    APP_ERROR_CHECK(err_code);
}


ret_code_t nfc_ble_pair_start(void)
{
    ret_code_t err_code;

    err_code = nfc_t2t_emulation_start();


    return err_code;
}


ret_code_t nfc_ble_pair_stop(void)
{
    ret_code_t err_code;

    err_code = nfc_t2t_emulation_stop();

    return err_code;
}


